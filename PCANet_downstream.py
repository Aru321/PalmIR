import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from collections import defaultdict


class PalmPCAIndexer:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.feature_db = defaultdict(list)  # 存储每个身份的特征
        self.identity_info = {}  # 存储身份到文件名的映射
        self.is_fitted = False

    def preprocess_image(self, image_path, img_size=(128, 128)):
        """预处理图像：灰度化、调整大小、归一化"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
        if img is None:
            print(f"Warning: Cannot read image {image_path}")
            return None

        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        return img.flatten()  # 展平为一维向量

    def build_database(self, database_dir):
        """构建PCA特征数据库"""
        print("Collecting images and extracting features...")

        all_features = []
        all_paths = []

        # 第一遍：收集所有图像数据
        for root, dirs, files in os.walk(database_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    feature = self.preprocess_image(img_path)

                    if feature is not None:
                        all_features.append(feature)
                        all_paths.append(img_path)

        if not all_features:
            raise ValueError("No valid images found in the database directory")

        # 转换为numpy数组
        X = np.array(all_features)
        print(f"Collected {len(X)} images with feature dimension: {X.shape}")

        # 标准化并训练PCA
        print("Fitting PCA...")
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

        # 第二遍：为每个图像生成PCA特征并分组到身份
        print("Building feature database...")
        for i, img_path in enumerate(all_paths):
            feature = all_features[i]
            feature_scaled = self.scaler.transform([feature])[0]
            pca_feature = self.pca.transform([feature_scaled])[0]

            # 从文件名提取身份信息
            filename = os.path.basename(img_path)
            identity = self._extract_identity(filename)

            # 存储特征和路径
            self.feature_db[identity].append({
                'feature': pca_feature,
                'filepath': img_path
            })

            # 存储身份信息
            if identity not in self.identity_info:
                self.identity_info[identity] = filename.split('_')[:2]  # 取数据集和ID部分

        self.is_fitted = True
        print(f"Database built with {len(self.feature_db)} identities")
        print(f"PCA feature dimension: {self.n_components}")

    def _extract_identity(self, filename):
        """从文件名提取身份信息，如 'CASIA_001_XXXX.jpg' -> 'CASIA_001'"""
        # 去掉文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        # 取前两部分作为身份标识
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        else:
            return name_without_ext

    def save_database(self, save_path):
        """保存特征数据库"""
        if not self.is_fitted:
            raise ValueError("Database not built yet")

        with open(save_path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'scaler': self.scaler,
                'feature_db': dict(self.feature_db),
                'identity_info': self.identity_info,
                'n_components': self.n_components
            }, f)
        print(f"Database saved to {save_path}")

    def load_database(self, load_path):
        """加载特征数据库"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.pca = data['pca']
        self.scaler = data['scaler']
        self.feature_db = defaultdict(list, data['feature_db'])
        self.identity_info = data['identity_info']
        self.n_components = data['n_components']
        self.is_fitted = True
        print(f"Database loaded with {len(self.feature_db)} identities")


class PalmPCASearcher:
    def __init__(self, indexer,exclude_self=False):
        self.indexer = indexer
        self.exclude_self = exclude_self
    def extract_query_feature(self, query_image_path):
        """提取查询图像的PCA特征"""
        feature = self.indexer.preprocess_image(query_image_path)
        if feature is None:
            raise ValueError(f"Cannot process query image: {query_image_path}")

        feature_scaled = self.indexer.scaler.transform([feature])[0]
        pca_feature = self.indexer.pca.transform([feature_scaled])[0]
        return pca_feature

    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query_image_path, top_k=5, strategy='average'):
        """
        搜索最相似的身份

        Args:
            query_image_path: 查询图片路径
            top_k: 返回前K个结果
            strategy: 'average' - 平均相似度, 'max' - 最大相似度
        """
        if not self.indexer.is_fitted:
            raise ValueError("PCA indexer not fitted yet")

        # 提取查询特征
        query_feature = self.extract_query_feature(query_image_path)
        # 提取查询图片的文件名（用于排除自己）
        query_filename = os.path.basename(query_image_path)
        scores = {}

        # 计算与每个身份的相似度
        for identity, features_list in self.indexer.feature_db.items():
            identity_scores = []

            for feature_info in features_list:
                db_feature = feature_info['feature']
                # 排除自己，否则会检索到自己
                db_filename = os.path.basename(feature_info['filepath'])
                if self.exclude_self and db_filename == query_filename:
                    continue
                #
                similarity = self.cosine_similarity(query_feature, db_feature)
                identity_scores.append(similarity)

            # 根据策略聚合分数
            if strategy == 'average':
                score = np.mean(identity_scores)
            elif strategy == 'max':
                score = np.max(identity_scores)
            else:
                raise ValueError("Strategy must be 'average' or 'max'")

            scores[identity] = {
                'score': score,
                'num_samples': len(identity_scores)
            }

        # 按分数排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)

        return sorted_results[:top_k]

    def print_results(self, results):
        """格式化打印结果"""
        print("\n=== Search Results ===")
        for rank, (identity, info) in enumerate(results, 1):
            dataset, id_num = identity.split('_')
            print(f"Rank {rank}: {dataset}_{id_num} | "
                  f"Score: {info['score']:.4f} | "
                  f"Samples: {info['num_samples']}")


def main():
    # 初始化索引器
    indexer = PalmPCAIndexer(n_components=50)  # 降维到50维

    # 构建数据库（只需要运行一次）
    database_dir = r'../datasets/Single_Dataset_Exp/IITD/test'
    indexer.build_database(database_dir)
    indexer.save_database("./pca_database/palm_IITD_pca_database.pkl")

    database_dir = r'../datasets/Single_Dataset_Exp/Tongji/test'
    indexer.build_database(database_dir)
    indexer.save_database("./pca_database/palm_Tongji_pca_database.pkl")

    # # # 或者直接加载已有的数据库
    # indexer.load_database("./pca_database/palm_pca_database.pkl")
    # #
    # # 初始化搜索器
    # searcher = PalmPCASearcher(indexer)
    #
    # # 执行查询
    # query_image = "/path/to/your/query_image.jpg"
    # results = searcher.search(query_image, top_k=5, strategy='average')
    #
    # # 显示结果
    # searcher.print_results(results)




if __name__ == "__main__":
    main()