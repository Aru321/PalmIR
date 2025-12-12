import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BatchPalmEvaluator:
    def __init__(self, searcher):
        self.searcher = searcher

    def extract_identity_from_filename(self, filename):
        """从文件名提取身份信息"""
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        else:
            return name_without_ext

    def batch_search(self, query_folder, top_k=1, strategy='average'):
        """
        批量查询文件夹中的所有图片

        Args:
            query_folder: 查询图片文件夹路径
            top_k: 返回前K个结果
            strategy: 相似度计算策略

        Returns:
            results: 包含所有查询结果的列表
        """
        results = []
        query_files = []

        # 收集所有查询图片
        for file in os.listdir(query_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                query_files.append(file)

        print(f"Found {len(query_files)} query images in {query_folder}")

        # 逐个查询
        for i, query_file in enumerate(query_files):
            query_path = os.path.join(query_folder, query_file)
            true_identity = self.extract_identity_from_filename(query_file)

            try:
                # 执行查询
                search_results = self.searcher.search(query_path, top_k=top_k, strategy=strategy)

                # 提取预测的身份（Top-1结果）
                if search_results:
                    pred_identity = search_results[0][0]  # Top-1预测身份
                    pred_score = search_results[0][1]['score']  # 预测分数

                    # 提取Top-K的所有预测身份
                    top_k_identities = [result[0] for result in search_results]
                else:
                    pred_identity = "unknown"
                    pred_score = 0.0
                    top_k_identities = []

                # 判断是否正确
                is_correct = (true_identity in top_k_identities)

                results.append({
                    'query_file': query_file,
                    'true_identity': true_identity,
                    'pred_identity': pred_identity,
                    'pred_score': pred_score,
                    'is_correct': is_correct,
                    'top_k_identities': top_k_identities,
                    'search_results': search_results
                })

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(query_files)} images...")

            except Exception as e:
                print(f"Error processing {query_file}: {e}")
                results.append({
                    'query_file': query_file,
                    'true_identity': true_identity,
                    'pred_identity': 'error',
                    'pred_score': 0.0,
                    'is_correct': False,
                    'top_k_identities': [],
                    'search_results': []
                })

        return results

    def calculate_accuracy(self, results, top_k=1):
        """
        计算准确率

        Args:
            results: batch_search返回的结果
            top_k: Top-K准确率

        Returns:
            metrics: 各种准确率指标
        """
        if not results:
            return {}

        # Top-1准确率
        correct_top1 = sum(1 for r in results if r['is_correct'])
        accuracy_top1 = correct_top1 / len(results)

        # Top-K准确率
        correct_topk = 0
        for r in results:
            if r['true_identity'] in r['top_k_identities'][:top_k]:
                correct_topk += 1
        accuracy_topk = correct_topk / len(results)

        # 按身份统计
        identity_stats = {}
        for r in results:
            identity = r['true_identity']
            if identity not in identity_stats:
                identity_stats[identity] = {'total': 0, 'correct': 0}

            identity_stats[identity]['total'] += 1
            if r['is_correct']:
                identity_stats[identity]['correct'] += 1

        # 计算每个身份的准确率
        identity_accuracy = {}
        for identity, stats in identity_stats.items():
            identity_accuracy[identity] = stats['correct'] / stats['total']

        metrics = {
            'total_queries': len(results),
            'accuracy_top1': accuracy_top1,
            'accuracy_topk': accuracy_topk,
            'correct_top1': correct_top1,
            'correct_topk': correct_topk,
            'identity_stats': identity_stats,
            'identity_accuracy': identity_accuracy
        }

        return metrics

    def generate_report(self, results, save_path=None):
        """生成详细的评估报告"""
        metrics = self.calculate_accuracy(results)

        print("\n" + "=" * 60)
        print("BATCH SEARCH EVALUATION REPORT")
        print("=" * 60)

        print(f"\nOverall Performance:")
        print(f"Total Query Images: {metrics['total_queries']}")
        print(f"Top-1 Accuracy: {metrics['accuracy_top1']:.4f} ({metrics['correct_top1']}/{metrics['total_queries']})")
        print(
            f"Top-{len(results[0]['top_k_identities']) if results else 0} Accuracy: {metrics['accuracy_topk']:.4f} ({metrics['correct_topk']}/{metrics['total_queries']})")

        print(f"\nPer-Identity Accuracy:")
        for identity, acc in sorted(metrics['identity_accuracy'].items()):
            stats = metrics['identity_stats'][identity]
            print(f"  {identity}: {acc:.4f} ({stats['correct']}/{stats['total']})")

        # 显示一些错误案例
        errors = [r for r in results if not r['is_correct'] and r['pred_identity'] != 'error']
        if errors:
            print(f"\nError Cases (showing first 5):")
            for i, error in enumerate(errors[:5]):
                print(
                    f"  {error['query_file']}: True={error['true_identity']}, Pred={error['pred_identity']}, Score={error['pred_score']:.4f}")

        # 保存详细结果到CSV
        if save_path:
            self.save_detailed_results(results, save_path)

        return metrics

    def save_detailed_results(self, results, save_path):
        """保存详细结果到CSV文件"""
        df_data = []
        for r in results:
            df_data.append({
                'query_file': r['query_file'],
                'true_identity': r['true_identity'],
                'pred_identity': r['pred_identity'],
                'pred_score': r['pred_score'],
                'is_correct': r['is_correct'],
                'top_k_predictions': ', '.join(r['top_k_identities'])
            })

        df = pd.DataFrame(df_data)
        df.to_csv(save_path, index=False)
        print(f"\nDetailed results saved to: {save_path}")

    def plot_confusion_matrix(self, results, save_path=None):
        """绘制混淆矩阵"""
        if not results:
            print("No results to plot")
            return

        # 提取所有身份
        all_identities = sorted(set([r['true_identity'] for r in results] +
                                    [r['pred_identity'] for r in results if r['pred_identity'] != 'error']))

        # 创建标签映射
        identity_to_idx = {identity: idx for idx, identity in enumerate(all_identities)}

        # 准备混淆矩阵数据
        y_true = [identity_to_idx[r['true_identity']] for r in results]
        y_pred = [identity_to_idx.get(r['pred_identity'], -1) for r in results]

        # 过滤掉错误预测
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]

        # 计算混淆矩阵
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(all_identities)))

        # 绘制
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_identities, yticklabels=all_identities)
        plt.title('Confusion Matrix - Palm Print Recognition')
        plt.xlabel('Predicted Identity')
        plt.ylabel('True Identity')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()


# def main(query_folder):
#     from PCANet_downstream import PalmPCAIndexer
#     from PCANet_downstream import PalmPCASearcher
#     # 初始化索引器和搜索器（使用之前定义的类）
#     indexer = PalmPCAIndexer(n_components=50)
#
#     # 加载已有的PCA数据库
#     indexer.load_database(r"./pca_database/palm_pca_database.pkl")
#     searcher = PalmPCASearcher(indexer,exclude_self=True)
#
#     # 初始化评估器
#     evaluator = BatchPalmEvaluator(searcher)
#
#     # 设置查询文件夹路径
#     query_folder = query_folder
#
#     print("Starting batch evaluation...")
#
#     # 执行批量查询
#     results = evaluator.batch_search(
#         query_folder=query_folder,
#         top_k=3,  # 返回Top-3结果
#         strategy='max'  # 使用平均相似度
#     )
#
#     # 生成评估报告
#     metrics = evaluator.generate_report(
#         results,
#         save_path="detailed_results.csv"
#     )

    # # 绘制混淆矩阵
    # evaluator.plot_confusion_matrix(
    #     results,
    #     save_path="confusion_matrix.png"
    # )

    # # 额外的统计分析
    # print("\nAdditional Statistics:")
    # scores = [r['pred_score'] for r in results if r['is_correct']]
    # if scores:
    #     print(f"Average score for correct predictions: {np.mean(scores):.4f}")
    #     print(f"Score std for correct predictions: {np.std(scores):.4f}")

    # # 按数据集分析（如果包含多个数据集）
    # datasets = {}
    # for r in results:
    #     dataset = r['true_identity'].split('_')[0]
    #     if dataset not in datasets:
    #         datasets[dataset] = {'total': 0, 'correct': 0}
    #     datasets[dataset]['total'] += 1
    #     if r['is_correct']:
    #         datasets[dataset]['correct'] += 1
    #
    # print(f"\nPerformance by Dataset:")
    # for dataset, stats in datasets.items():
    #     accuracy = stats['correct'] / stats['total']
    #     print(f"  {dataset}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")

# 📊 Results by Degradation: PalmIR Large
#   gaussian: 0.9868
#   poisson: 0.9901
#   motion_blur: 0.9474
#   low_light: 0.7039
#   inpaint: 0.9901
#   mixed: 0.9704

def main():
    from PCANet_downstream import PalmPCAIndexer
    from PCANet_downstream import PalmPCASearcher

    # 方法列表和退化类型
    methods = [
        'PalmIR',
        'UNet',
               'GUNet',
        'SOTA_DHA',
        'VIRNet',
        'AdaIR',
            ]
    degradations = [
                    'gaussian',
                    'poisson',
                    'motion_blur',
                    'low_light',
                     'inpaint',
                     'mixed']

    # 基础路径
    base_path = r"../datasets/Validation/IITD_and_Unified"

    # 存储所有结果的字典
    all_results = {}

    # 初始化索引器和搜索器
    indexer = PalmPCAIndexer(n_components=50)
    indexer.load_database(r"./pca_database/palm_IITD_pca_database.pkl")
    searcher = PalmPCASearcher(indexer,exclude_self=True)
    evaluator = BatchPalmEvaluator(searcher)

    print("Starting batch evaluation for all methods and degradations...")

    # 遍历所有方法和退化组合
    for method in methods:
        for degradation in degradations:
            # 构建查询文件夹路径
            query_folder = f"{base_path}/{method}_{degradation}"

            # 检查文件夹是否存在
            if not os.path.exists(query_folder):
                print(f"⚠️  Folder not found: {query_folder}")
                continue

            print(f"\n{'=' * 60}")
            print(f"Testing: {method}_{degradation}")
            print(f"Folder: {query_folder}")
            print(f"{'=' * 60}")

            try:
                # 执行批量查询
                results = evaluator.batch_search(
                    query_folder=query_folder,
                    top_k=3,
                    strategy='max',
                )

                # 计算Top-1准确率
                correct_top1 = sum(1 for r in results if r['is_correct'])
                total_queries = len(results)
                accuracy_top1 = correct_top1 / total_queries if total_queries > 0 else 0

                # 存储结果
                all_results[f"{method}_{degradation}"] = {
                    'accuracy': accuracy_top1,
                    'correct': correct_top1,
                    'total': total_queries,
                    'folder': query_folder
                }

                print(f"✅ Top-1 Accuracy: {accuracy_top1:.4f} ({correct_top1}/{total_queries})")

            except Exception as e:
                print(f"❌ Error processing {method}_{degradation}: {e}")
                all_results[f"{method}_{degradation}"] = {
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0,
                    'folder': query_folder,
                    'error': str(e)
                }

    # 生成汇总报告
    print(f"\n{'=' * 80}")
    print("SUMMARY REPORT")
    print(f"{'=' * 80}")

    # 按方法汇总
    print("\n📊 Results by Method:")
    method_accuracies = {}
    for method in methods:
        method_results = [v for k, v in all_results.items() if k.startswith(method)]
        if method_results:
            avg_accuracy = np.mean([r['accuracy'] for r in method_results])
            method_accuracies[method] = avg_accuracy
            print(f"  {method}: {avg_accuracy:.4f}")

    # 按退化类型汇总
    print("\n📊 Results by Degradation:")
    degradation_accuracies = {}
    for degradation in degradations:
        degradation_results = [v for k, v in all_results.items() if k.endswith(degradation)]
        if degradation_results:
            avg_accuracy = np.mean([r['accuracy'] for r in degradation_results])
            degradation_accuracies[degradation] = avg_accuracy
            print(f"  {degradation}: {avg_accuracy:.4f}")

    # 详细结果表格
    print(f"\n{'=' * 80}")
    print("DETAILED RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Method_Degradation':<25} {'Accuracy':<10} {'Correct/Total':<15} {'Folder'}")
    print(f"{'-' * 80}")

    for key, result in sorted(all_results.items()):
        if 'error' in result:
            print(f"{key:<25} {'ERROR':<10} {'-':<15} {result['folder']}")
        else:
            print(f"{key:<25} {result['accuracy']:.4f}    {result['correct']}/{result['total']:<12} {result['folder']}")

    # 保存结果到文件
    save_results_to_csv(all_results, "palm_recognition_results.csv")

    return all_results


def save_results_to_csv(results, filename):
    """保存结果到CSV文件"""
    import pandas as pd

    data = []
    for key, value in results.items():
        if 'error' in value:
            row = {
                'method_degradation': key,
                'accuracy': 0,
                'correct': 0,
                'total': 0,
                'folder': value['folder'],
                'error': value['error']
            }
        else:
            row = {
                'method_degradation': key,
                'accuracy': value['accuracy'],
                'correct': value['correct'],
                'total': value['total'],
                'folder': value['folder'],
                'error': ''
            }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")


# 如果你还想保留单个文件夹测试的功能
def test_single_folder(query_folder):
    """测试单个文件夹"""
    from PCANet_downstream import PalmPCAIndexer
    from PCANet_downstream import PalmPCASearcher

    indexer = PalmPCAIndexer(n_components=50)
    indexer.load_database(r"./pca_database/palm_pca_database.pkl")
    searcher = PalmPCASearcher(indexer,exclude_self=True)
    evaluator = BatchPalmEvaluator(searcher)

    print(f"Testing single folder: {query_folder}")

    results = evaluator.batch_search(
        query_folder=query_folder,
        top_k=3,
        strategy='max',
    )

    # 计算Top-1准确率
    correct_top1 = sum(1 for r in results if r['is_correct'])
    total_queries = len(results)
    accuracy_top1 = correct_top1 / total_queries if total_queries > 0 else 0

    print(f"Top-1 Accuracy: {accuracy_top1:.4f} ({correct_top1}/{total_queries})")

    return accuracy_top1, results


if __name__ == "__main__":
    # 运行批量测试
    all_results = main()

    # 如果只想测试单个文件夹，取消下面的注释
    # query_folder = r"../datasets/Validation/IITD_and_Unified/"
    # accuracy, results = test_single_folder(query_folder)

    # 测试退化
    # query_folders = {
    #     # r"../datasets/single_degraded/Low_light",
    #                  # r"../datasets/single_degraded/Inpaint",
    #                  # r"../datasets/single_degraded/Mixed",
    #                  r"../datasets/single_degraded/Gaussian",
    #                 # r"../datasets/single_degraded/Poisson/",
    #                 # r"../datasets/single_degraded/Motion",
    #                  }

    # all_acc = []
    # for folder in query_folders:
    #
    #     accuracy, results = test_single_folder(folder)
    #     print(accuracy)
    #     all_acc.append(accuracy)
    #
    # print(all_acc)
# if __name__ == "__main__":
#     main(
#         query_folder=r"../datasets/Validation/Unified/AdaIR_gaussian"
#     )