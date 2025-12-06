"""
Type 1 Feature Noise Generator with Decision Boundary Constraint (Fast Version)

이 스크립트는 Type 1 feature noise를 미리 생성해서 파일로 저장합니다.
생성된 노이즈는 다음 두 조건을 모두 만족합니다:
1. Mahalanobis 거리 기준 99% 밖에 위치 (빠른 Cholesky + radial scaling 방식)
2. Base SVM의 decision boundary를 넘음 (잘못 분류됨)

사용법:
    python make_noise_fast.py [--seed 42] [--ratios 0.05 0.10 0.15 0.20]
    
자동으로 현재 디렉토리의 하위 폴더에서 .npz 파일을 찾아 처리합니다.
"""

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import argparse
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def precompute_class_stats(X_train_scaled, y_train, percentile=99, categorical_mask=None):
    """
    각 클래스별로 통계량을 미리 계산 (빠른 버전 - Cholesky 사용)
    
    Args:
        X_train_scaled: StandardScaler로 스케일된 학습 데이터
        y_train: 레이블
        percentile: Mahalanobis 거리 percentile threshold
        categorical_mask: 카테고리 feature 마스크 (True: 카테고리, False: 연속형)
    
    Returns:
        class_stats: 클래스별 통계량 딕셔너리
    """
    class_stats = {}
    classes = np.unique(y_train)
    
    # 연속형 feature만 선택
    if categorical_mask is not None:
        continuous_mask = ~categorical_mask
    else:
        continuous_mask = np.ones(X_train_scaled.shape[1], dtype=bool)
    
    for cls in classes:
        X_cls = X_train_scaled[y_train == cls]
        
        # 연속형 feature에 대해서만 통계량 계산
        X_cls_continuous = X_cls[:, continuous_mask]
        
        mu = X_cls_continuous.mean(axis=0)
        cov = np.cov(X_cls_continuous.T) + np.eye(X_cls_continuous.shape[1]) * 1e-6
        
        # Cholesky 분해 추가 (빠른 생성용)
        L = np.linalg.cholesky(cov)
        
        # Mahalanobis 거리 계산용
        inv_cov = np.linalg.inv(cov)
        diffs = X_cls_continuous - mu
        dists = np.einsum('ni,ij,nj->n', diffs, inv_cov, diffs)
        tau = np.percentile(dists, percentile)
        
        # 카테고리 feature의 평균값도 저장 (복사용)
        if categorical_mask is not None:
            mu_categorical = X_cls[:, categorical_mask].mean(axis=0)
        else:
            mu_categorical = None
        
        class_stats[cls] = {
            "mu": mu,
            "L": L,  # Cholesky factor 추가
            "tau": tau,
            "mu_categorical": mu_categorical,
        }
    
    return class_stats


def generate_type1_noise_with_boundary(
    X_train, y_train, 
    ratios=[0.05, 0.10, 0.15, 0.20],
    random_state=42,
    output_dir=".",
    dataset_name="dataset",
    visualize=True,
    categorical_mask=None
):
    """
    Type 1 feature noise를 생성하고 각 ratio별로 파일로 저장 (빠른 버전)
    
    Args:
        X_train: 원본 학습 데이터 (N, d)
        y_train: 원본 레이블 (N,)
        ratios: 노이즈 비율 리스트 (예: [0.05, 0.10, 0.15, 0.20])
        random_state: 랜덤 시드
        output_dir: 저장 디렉토리
        dataset_name: 데이터셋 이름
        visualize: 시각화 수행 여부
        categorical_mask: 카테고리 feature 마스크 (True: 카테고리, False: 연속형)
    """
    rng = default_rng(random_state)
    N = len(X_train)
    d = X_train.shape[1]
    
    # 카테고리 feature 정보 출력
    if categorical_mask is not None:
        n_categorical = np.sum(categorical_mask)
        n_continuous = d - n_categorical
        print(f"원본 데이터: N={N}, d={d} (연속형: {n_continuous}, 카테고리: {n_categorical})")
        print(f"카테고리 feature는 노이즈에서 제외됩니다.")
    else:
        print(f"원본 데이터: N={N}, d={d}")
    print(f"목표 노이즈 비율: {[f'{r*100:.0f}%' for r in ratios]}")
    print("-" * 60)
    
    # 1. StandardScaler로 스케일링
    print("1. 데이터 스케일링...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 2. Base SVM 학습 (decision boundary 용)
    print("2. Base SVM 학습 중...")
    base_svm = LinearSVC(C=1.0, max_iter=10000, random_state=random_state, dual='auto')
    base_svm.fit(X_train_scaled, y_train)
    train_acc = base_svm.score(X_train_scaled, y_train)
    print(f"   Base SVM 학습 완료 (Train Accuracy: {train_acc:.4f})")
    
    # 3. 클래스별 통계량 계산
    print("3. 클래스별 통계량 계산...")
    class_stats = precompute_class_stats(X_train_scaled, y_train, percentile=99, categorical_mask=categorical_mask)
    classes = np.array(list(class_stats.keys()))
    print(f"   클래스 개수: {len(classes)}")
    
    # 4. Type 1 노이즈 생성
    print("\n4. Type 1 노이즈 생성 시작 (빠른 방식: Cholesky + radial scaling)...")
    print("   조건: ① Mahalanobis 99% 밖 + ② Decision boundary 넘음")
    print("   (추가: boundary 안 넘은 것도 함께 저장)")
    print("-" * 60)
    
    # boundary 넘은 노이즈 (메인)
    crossed_outliers_X = []
    crossed_outliers_y = []
    crossed_predictions = []
    
    # boundary 안 넘은 노이즈 (부가)
    not_crossed_outliers_X = []
    not_crossed_outliers_y = []
    not_crossed_predictions = []
    
    max_ratio = max(ratios)
    max_outliers = int(N * max_ratio)  # boundary 넘은 것 기준
    
    attempt_count = 0
    
    # 각 ratio에 대한 저장 완료 여부 추적
    saved_ratios = set()
    
    while len(crossed_outliers_X) < max_outliers:
        # 클래스 선택 (균등 확률)
        cls = rng.choice(classes)
        stats = class_stats[cls]
        mu = stats["mu"]
        L = stats["L"]
        tau = stats["tau"]
        mu_categorical = stats["mu_categorical"]
        
        # 연속형 feature 차원 수
        if categorical_mask is not None:
            d_continuous = np.sum(~categorical_mask)
        else:
            d_continuous = d
        
        # 빠른 방식: Cholesky + radial scaling으로 Mahalanobis 99% 밖 샘플 생성
        z = rng.standard_normal(size=d_continuous)
        
        mah_distance = np.dot(z, z)
        E = rng.exponential(scale=2 / tau)
        r2 = tau + E
        
        scale = np.sqrt(r2 / (mah_distance + 1e-12))
        z = z * scale
        
        # 연속형 feature에만 노이즈 적용
        x_candidate_continuous = mu + L @ z
        
        # 전체 feature 벡터 구성
        if categorical_mask is not None:
            x_candidate_scaled = np.zeros(d)
            x_candidate_scaled[~categorical_mask] = x_candidate_continuous
            # 카테고리 feature는 해당 클래스의 평균값 사용 (반올림하여 0 or 1)
            x_candidate_scaled[categorical_mask] = np.round(mu_categorical)
        else:
            x_candidate_scaled = x_candidate_continuous
        
        attempt_count += 1
        
        # 조건 ②: Decision boundary를 넘는지 확인
        pred_label = base_svm.predict(x_candidate_scaled.reshape(1, -1))[0]
        
        # Boundary 넘은 것과 안 넘은 것을 구분하여 저장
        if pred_label != cls:
            # Boundary 넘음 - 메인 카운트
            crossed_outliers_X.append(x_candidate_scaled)
            crossed_outliers_y.append(cls)
            crossed_predictions.append(pred_label)
        else:
            # Boundary 안 넘음 - 부가 정보로 저장
            not_crossed_outliers_X.append(x_candidate_scaled)
            not_crossed_outliers_y.append(cls)
            not_crossed_predictions.append(pred_label)
        
        current_count = len(crossed_outliers_X)
        
        # 진행 상황 출력 (100개마다)
        if current_count % 100 == 0 or current_count in [int(N * r) for r in ratios]:
            accept_rate = current_count / attempt_count if attempt_count > 0 else 0
            total_generated = len(crossed_outliers_X) + len(not_crossed_outliers_X)
            print(f"   Boundary 넘음: {current_count}/{max_outliers} "
                  f"(전체 생성: {total_generated}, 수락률: {accept_rate:.2%})")
        
        # 각 ratio에 도달할 때마다 저장
        for ratio in ratios:
            target_count = int(N * ratio)
            if current_count >= target_count and ratio not in saved_ratios:
                # 현재 시점의 전체 노이즈 스냅샷 (crossed + not_crossed)
                all_outliers_X = crossed_outliers_X[:target_count] + not_crossed_outliers_X[:]
                all_outliers_y = crossed_outliers_y[:target_count] + not_crossed_outliers_y[:]
                
                save_dataset(
                    X_train, y_train,
                    all_outliers_X,
                    all_outliers_y,
                    scaler,
                    ratio,
                    output_dir,
                    dataset_name
                )
                saved_ratios.add(ratio)
    
    print("-" * 60)
    print(f"\n최종 통계:")
    print(f"  - 총 시도: {attempt_count}")
    print(f"  - Boundary 넘음 (메인): {len(crossed_outliers_X)}")
    print(f"  - Boundary 안넘음 (부가): {len(not_crossed_outliers_X)}")
    print(f"  - 전체 생성: {len(crossed_outliers_X) + len(not_crossed_outliers_X)}")
    
    total_accepted = len(crossed_outliers_X) + len(not_crossed_outliers_X)
    if total_accepted > 0:
        cross_rate = len(crossed_outliers_X) / total_accepted
        print(f"  - Boundary 통과율: {cross_rate:.2%}")
    
    print(f"\n저장 완료된 비율: {sorted([f'{r*100:.0f}%' for r in saved_ratios])}")
    
    # 시각화
    if visualize:
        print("\n5. 시각화 생성 중...")
        vis_dir = Path(output_dir) / "fast_vis"
        vis_dir.mkdir(exist_ok=True)
        
        # 각 ratio별로 시각화
        for idx, ratio in enumerate(ratios):
            target_count = int(N * ratio)
            
            # 해당 ratio 시점의 전체 노이즈 (crossed + not_crossed)
            # 각 ratio마다 crossed 개수는 다르지만, not_crossed는 최종 전체 사용
            all_outliers_X = crossed_outliers_X[:target_count] + not_crossed_outliers_X[:]
            all_outliers_y = crossed_outliers_y[:target_count] + not_crossed_outliers_y[:]
            all_predictions = crossed_predictions[:target_count] + not_crossed_predictions[:]
            
            # 전체 시각화
            visualize_noise_generation(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                base_svm=base_svm,
                scaler=scaler,
                accepted_outliers_X=all_outliers_X,
                accepted_outliers_y=all_outliers_y,
                accepted_predictions=all_predictions,
                ratio=ratio,
                vis_dir=vis_dir,
                dataset_name=dataset_name
            )
            
            # 클래스별 개별 시각화
            visualize_per_class(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                base_svm=base_svm,
                scaler=scaler,
                accepted_outliers_X=all_outliers_X,
                accepted_outliers_y=all_outliers_y,
                accepted_predictions=all_predictions,
                ratio=ratio,
                vis_dir=vis_dir,
                dataset_name=dataset_name
            )
        print(f"   시각화 저장 완료: {vis_dir}")


def save_dataset(
    X_train_orig, y_train_orig,
    outliers_X_scaled, outliers_y,
    scaler,
    ratio,
    output_dir,
    dataset_name
):
    """
    노이즈가 추가된 데이터셋을 저장
    
    Args:
        X_train_orig: 원본 학습 데이터 (스케일 전)
        y_train_orig: 원본 레이블
        outliers_X_scaled: 생성된 노이즈 (스케일된 공간)
        outliers_y: 노이즈 레이블
        scaler: StandardScaler 인스턴스
        ratio: 노이즈 비율
        output_dir: 저장 디렉토리
        dataset_name: 데이터셋 이름
    """
    # 스케일된 공간의 노이즈를 원래 공간으로 변환
    outliers_X_scaled_array = np.array(outliers_X_scaled)
    outliers_y_array = np.array(outliers_y)
    
    outliers_X_orig = scaler.inverse_transform(outliers_X_scaled_array)
    
    """ 노이즈만 저장하도록 수정 """
    # # 원본 데이터와 노이즈 결합
    # X_combined = np.vstack([X_train_orig, outliers_X_orig])
    # y_combined = np.concatenate([y_train_orig, outliers_y_array])
    
    # 파일 저장 (fast 버전임을 표시)
    output_path = Path(output_dir) / f"fast_{dataset_name}_type1_boundary_{int(ratio*100)}pct.npz"
    np.savez(output_path, X_train=outliers_X_orig, y_train=outliers_y_array)
    
    print(f"\n✓ 저장 완료: {output_path.name}")
    print(f"  - 원본: {len(X_train_orig)}, 노이즈: {len(outliers_X_orig)}, "
          f"{ratio*100:.0f}% 노이즈")


def find_dataset_files(base_dir="."):
    """
    현재 디렉토리의 하위 폴더에서 .npz 파일을 찾음
    
    Args:
        base_dir: 검색 시작 디렉토리
        
    Returns:
        list of tuples: (데이터셋 이름, .npz 파일 경로)
    """
    datasets = []
    
    # 하위 디렉토리 탐색
    for item in Path(base_dir).iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # 해당 디렉토리 내의 .npz 파일 찾기
            npz_files = list(item.glob("*.npz"))
            
            # type1_boundary가 포함된 파일은 제외 (이미 생성된 노이즈 파일)
            original_files = [f for f in npz_files if 'type1_boundary' not in f.name and not f.name.startswith('fast_')]
            
            if original_files:
                # 첫 번째 원본 파일 사용
                dataset_name = item.name
                npz_path = original_files[0]
                datasets.append((dataset_name, npz_path))
    
    return datasets


def visualize_noise_generation(
    X_train_scaled, y_train, base_svm, scaler,
    accepted_outliers_X, accepted_outliers_y, accepted_predictions,
    ratio, vis_dir, dataset_name
):
    """
    노이즈 생성 결과를 시각화
    
    Args:
        X_train_scaled: 스케일된 원본 데이터
        y_train: 원본 레이블
        base_svm: 학습된 SVM 모델
        scaler: StandardScaler 인스턴스
        accepted_outliers_X: 생성된 노이즈 (스케일된 공간)
        accepted_outliers_y: 노이즈의 원래 클래스
        accepted_predictions: 노이즈의 예측 클래스
        ratio: 노이즈 비율
        vis_dir: 시각화 저장 디렉토리
        dataset_name: 데이터셋 이름
    """
    # PCA로 2D 투영
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train_scaled)
    
    if len(accepted_outliers_X) > 0:
        outliers_2d = pca.transform(np.array(accepted_outliers_X))
    else:
        return
    
    # 클래스 개수
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    # 색상 설정
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # 그림 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== Left: Overall view (Original + All noise) ==========
    ax = axes[0]
    
    # Plot original data
    for idx, cls in enumerate(classes):
        mask = y_train == cls
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                  c=[colors[idx]], label=f'Class {cls} (Original)',
                  alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Plot noise points (color by original class)
    for idx, cls in enumerate(classes):
        mask = np.array(accepted_outliers_y) == cls
        if np.any(mask):
            ax.scatter(outliers_2d[mask, 0], outliers_2d[mask, 1],
                      c=[colors[idx]], marker='o', s=50,
                      label=f'Class {cls} (Noise)', 
                      edgecolors='red', linewidth=1.5, alpha=0.9)
    
    # Decision boundary 그리기 (PCA 공간에서)
    plot_decision_boundary_pca(ax, base_svm, pca, X_train_scaled)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'{dataset_name} - Type 1 Noise ({ratio*100:.0f}%)\nOriginal Data + Noise Points', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========== Right: Noise-only detail view ==========
    ax = axes[1]
    
    # Show original data in gray as background
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
              c='lightgray', alpha=0.3, s=30, label='Original Data')
    
    # 노이즈 포인트를 두 가지로 구분:
    # 1. Decision boundary를 넘은 것 (원래 클래스 != 예측 클래스)
    # 2. Decision boundary를 넘지 않은 것 (원래 클래스 == 예측 클래스)
    
    outliers_y_arr = np.array(accepted_outliers_y)
    predictions_arr = np.array(accepted_predictions)
    
    # Noise that crossed decision boundary (red circles)
    crossed_mask = outliers_y_arr != predictions_arr
    if np.any(crossed_mask):
        ax.scatter(outliers_2d[crossed_mask, 0], outliers_2d[crossed_mask, 1],
                  c='red', marker='o', s=60,
                  label=f'Crossed Boundary ({np.sum(crossed_mask)})',
                  edgecolors='darkred', linewidth=1.5, alpha=0.9, zorder=5)
    
    # Noise that didn't cross boundary (blue circles)
    not_crossed_mask = ~crossed_mask
    if np.any(not_crossed_mask):
        ax.scatter(outliers_2d[not_crossed_mask, 0], outliers_2d[not_crossed_mask, 1],
                  c='blue', marker='o', s=60,
                  label=f'Not Crossed ({np.sum(not_crossed_mask)})',
                  edgecolors='darkblue', linewidth=1.5, alpha=0.9, zorder=5)
    
    # Decision boundary 그리기
    plot_decision_boundary_pca(ax, base_svm, pca, X_train_scaled, alpha=0.4)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'Noise Detail Analysis\nGenerated Noise: {len(accepted_outliers_X)} points', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 전체 타이틀
    explained_var = pca.explained_variance_ratio_
    fig.suptitle(f'{dataset_name} - Type 1 Feature Noise Visualization ({ratio*100:.0f}%)\n'
                f'PCA explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 저장
    output_path = vis_dir / f"{dataset_name}_type1_noise_{int(ratio*100)}pct.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ {output_path.name}")


def plot_decision_boundary_pca(ax, svm, pca, X_train_scaled, alpha=0.2):
    """
    PCA로 투영된 2D 공간에서 decision boundary를 근사적으로 그림
    
    Args:
        ax: matplotlib axis
        svm: 학습된 SVM 모델
        pca: PCA 인스턴스
        X_train_scaled: 스케일된 원본 데이터 (boundary 범위 결정용)
        alpha: 투명도
    """
    # 2D PCA 공간에서 그리드 생성
    X_2d = pca.transform(X_train_scaled)
    
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    h = (x_max - x_min) / 200  # 해상도
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 2D 그리드를 원래 차원으로 역변환
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_2d)
    
    # SVM으로 예측
    Z = svm.predict(grid_original)
    Z = Z.reshape(xx.shape)
    
    # Decision boundary 그리기
    ax.contourf(xx, yy, Z, alpha=alpha, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=1.5, alpha=0.5)


def visualize_per_class(
    X_train_scaled, y_train, base_svm, scaler,
    accepted_outliers_X, accepted_outliers_y, accepted_predictions,
    ratio, vis_dir, dataset_name
):
    """
    클래스별로 개별 시각화 생성 (해당 클래스만 빨간색으로 강조)
    
    Args:
        X_train_scaled: 스케일된 원본 데이터
        y_train: 원본 레이블
        base_svm: 학습된 SVM 모델
        scaler: StandardScaler 인스턴스
        accepted_outliers_X: 생성된 노이즈 (스케일된 공간)
        accepted_outliers_y: 노이즈의 원래 클래스
        accepted_predictions: 노이즈의 예측 클래스
        ratio: 노이즈 비율
        vis_dir: 시각화 저장 디렉토리
        dataset_name: 데이터셋 이름
    """
    if len(accepted_outliers_X) == 0:
        return
    
    # PCA로 2D 투영
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train_scaled)
    outliers_2d = pca.transform(np.array(accepted_outliers_X))
    
    classes = np.unique(y_train)
    outliers_y_arr = np.array(accepted_outliers_y)
    predictions_arr = np.array(accepted_predictions)
    
    # 각 클래스별로 시각화
    for target_cls in classes:
        # 클래스별 폴더 생성
        class_dir = vis_dir / f"class{target_cls}"
        class_dir.mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 원본 데이터 중 target_cls가 아닌 것은 회색으로
        other_original_mask = y_train != target_cls
        ax.scatter(X_train_2d[other_original_mask, 0], X_train_2d[other_original_mask, 1],
                  c='lightgray', alpha=0.3, s=30, label='Other Classes (Original)')
        
        # 원본 데이터 중 target_cls는 연한 색으로 진하게
        target_original_mask = y_train == target_cls
        ax.scatter(X_train_2d[target_original_mask, 0], X_train_2d[target_original_mask, 1],
                  c='lightcoral', alpha=0.8, s=50, 
                  edgecolors='salmon', linewidth=1,
                  label=f'Class {target_cls} (Original)')
        
        # 노이즈 포인트 중 target_cls가 아닌 것은 회색으로
        other_mask = outliers_y_arr != target_cls
        if np.any(other_mask):
            ax.scatter(outliers_2d[other_mask, 0], outliers_2d[other_mask, 1],
                      c='gray', marker='o', s=50,
                      alpha=0.3, edgecolors='darkgray', linewidth=1)
        
        # target_cls에 해당하는 노이즈만 빨간색으로 강조
        target_mask = outliers_y_arr == target_cls
        if np.any(target_mask):
            # target_cls 노이즈의 예측 결과
            target_predictions = predictions_arr[target_mask]
            target_outliers_2d = outliers_2d[target_mask]
            
            # target_cls 중에서 boundary 넘은 것 (원래 클래스 != 예측 클래스)
            crossed_in_target = target_predictions != target_cls
            if np.any(crossed_in_target):
                ax.scatter(target_outliers_2d[crossed_in_target, 0], target_outliers_2d[crossed_in_target, 1],
                          c='red', marker='o', s=80,
                          label=f'Class {target_cls} - Crossed ({np.sum(crossed_in_target)})',
                          edgecolors='darkred', linewidth=2, alpha=0.95, zorder=5)
            
            # target_cls 중에서 boundary 안 넘은 것 (원래 클래스 == 예측 클래스)
            not_crossed_in_target = target_predictions == target_cls
            if np.any(not_crossed_in_target):
                ax.scatter(target_outliers_2d[not_crossed_in_target, 0], target_outliers_2d[not_crossed_in_target, 1],
                          c='orange', marker='o', s=80,
                          label=f'Class {target_cls} - Not Crossed ({np.sum(not_crossed_in_target)})',
                          edgecolors='darkorange', linewidth=2, alpha=0.95, zorder=5)
        
        # Decision boundary 그리기
        plot_decision_boundary_pca(ax, base_svm, pca, X_train_scaled, alpha=0.3)
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(f'{dataset_name} - Class {target_cls} Focus ({ratio*100:.0f}%)\n'
                    f'Red: Class {target_cls} noise, Gray: Other classes',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 클래스별 폴더에 저장
        output_path = class_dir / f"{dataset_name}_class{target_cls}_{int(ratio*100)}pct.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   ✓ Class-specific visualizations ({len(classes)} classes)")


def main():
    parser = argparse.ArgumentParser(
        description="Type 1 Feature Noise Generator with Decision Boundary Constraint (Fast Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python make_noise_fast.py
    python make_noise_fast.py --seed 42 --ratios 0.05 0.10 0.15 0.20
    
자동으로 하위 폴더(breast_cancer, iris_2feat, wine 등)에서 
.npz 파일을 찾아 각 폴더 내에 노이즈 데이터를 생성합니다.
        """
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
        help="노이즈 비율 리스트 (기본값: 0.05 0.10 0.15 0.20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="시각화 생성 안함"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Type 1 Feature Noise Generator (Fast Version)")
    print("=" * 60)
    print(f"설정: seed={args.seed}, ratios={args.ratios}\n")
    
    # 데이터셋 파일 찾기
    datasets = find_dataset_files(".")
    
    if not datasets:
        print("오류: 처리할 데이터셋을 찾을 수 없습니다.")
        print("현재 디렉토리의 하위 폴더에 .npz 파일이 있는지 확인하세요.")
        return
    
    print(f"발견된 데이터셋: {len(datasets)}개\n")
    
    # 각 데이터셋 처리
    for idx, (dataset_name, npz_path) in enumerate(datasets, 1):
        print("=" * 60)
        print(f"[{idx}/{len(datasets)}] 데이터셋: {dataset_name}")
        print("=" * 60)
        print(f"입력 파일: {npz_path}")
        
        try:
            # 데이터 로드
            data = np.load(npz_path)
            X_train = data['X_train']
            y_train = data['y_train']
            
            print(f"데이터 로드 완료: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # 노이즈 생성 및 저장 (같은 폴더에 저장)
            output_dir = npz_path.parent
            
            # Titanic 데이터셋의 경우 카테고리 feature 마스크 생성
            categorical_mask = None
            if dataset_name == "titanic":
                # Titanic features: Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S
                # 마지막 3개 feature가 one-hot encoded categorical features
                categorical_mask = np.zeros(X_train.shape[1], dtype=bool)
                categorical_mask[-3:] = True  # Sex_male, Embarked_Q, Embarked_S
                print(f"Titanic 데이터셋 감지: 카테고리 feature (indices {np.where(categorical_mask)[0].tolist()})는 노이즈에서 제외됩니다.")
            
            generate_type1_noise_with_boundary(
                X_train=X_train,
                y_train=y_train,
                ratios=args.ratios,
                random_state=args.seed,
                output_dir=output_dir,
                dataset_name=dataset_name,
                visualize=not args.no_vis,
                categorical_mask=categorical_mask
            )
            
            print(f"\n✓ {dataset_name} 완료!\n")
            
        except Exception as e:
            print(f"\n✗ {dataset_name} 처리 중 오류 발생: {e}\n")
            continue
    
    print("=" * 60)
    print("모든 데이터셋 처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
