#!/usr/bin/env python3
"""
演示脚本：展示NRI-MD项目在Python 3.10+环境中的基本功能

这个脚本演示了：
1. 数据处理功能
2. 模型创建和训练
3. 可视化功能
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 导入NRI-MD模块
from nri_md.utils import (
    my_softmax, gumbel_softmax, get_offdiag_indices, 
    encode_onehot, binary_accuracy
)
from nri_md.modules import MLP, MLPEncoder, MLPDecoder


def create_sample_data():
    """创建示例数据用于演示"""
    print("创建示例数据...")
    
    # 创建数据目录
    data_dir = Path("data")
    pdb_dir = data_dir / "pdb"
    data_dir.mkdir(exist_ok=True)
    pdb_dir.mkdir(exist_ok=True)
    
    # 创建示例PDB文件
    sample_pdb = pdb_dir / "demo.pdb"
    if not sample_pdb.exists():
        # 创建一个简单的3原子系统的轨迹
        pdb_content = ""
        for frame in range(10):  # 10帧
            for atom in range(3):  # 3个原子
                x = 20.0 + atom * 3.0 + frame * 0.1
                y = 17.0 + atom * 2.0 + frame * 0.05
                z = 10.0 + atom * 1.0 + frame * 0.02
                pdb_content += f"ATOM  {atom+1:5d}  CA  ALA A{atom+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            pdb_content += "ENDMDL\n"
        pdb_content += "END\n"
        
        with open(sample_pdb, 'w') as f:
            f.write(pdb_content)
        print(f"✓ 创建示例PDB文件: {sample_pdb}")
    
    # 创建示例numpy数据文件
    num_samples, num_atoms, timesteps, dims = 5, 3, 10, 6
    
    # 特征数据 (位置 + 速度)
    features = np.random.randn(num_samples, timesteps, dims, num_atoms)
    
    # 边数据 (连接矩阵)
    edges = np.random.randint(0, 2, (num_samples, num_atoms, num_atoms))
    # 移除自连接
    for i in range(num_atoms):
        edges[:, i, i] = 0
    
    # 保存数据
    np.save(data_dir / "features.npy", features)
    np.save(data_dir / "edges.npy", edges)
    np.save(data_dir / "features_valid.npy", features[:2])
    np.save(data_dir / "edges_valid.npy", edges[:2])
    np.save(data_dir / "features_test.npy", features[:1])
    np.save(data_dir / "edges_test.npy", edges[:1])
    
    print(f"✓ 创建示例数据文件，特征形状: {features.shape}")
    return features, edges


def test_utility_functions():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    # 测试softmax
    x = torch.randn(2, 3, 4)
    result = my_softmax(x, axis=1)
    assert torch.allclose(result.sum(dim=1), torch.ones(2, 4), atol=1e-6)
    print("✓ my_softmax 函数正常")
    
    # 测试Gumbel softmax
    logits = torch.randn(5, 3)
    result = gumbel_softmax(logits, tau=1.0, hard=False)
    assert torch.allclose(result.sum(dim=-1), torch.ones(5), atol=1e-6)
    print("✓ gumbel_softmax 函数正常")
    
    # 测试准确率计算
    output = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0, 1, 0, 1])
    acc = binary_accuracy(output, labels)
    assert acc == 1.0
    print("✓ binary_accuracy 函数正常")
    
    # 测试off-diagonal indices
    indices = get_offdiag_indices(4)
    assert len(indices) == 12  # 4*4 - 4 = 12
    print("✓ get_offdiag_indices 函数正常")


def test_neural_modules():
    """测试神经网络模块"""
    print("\n测试神经网络模块...")
    
    # 测试MLP
    mlp = MLP(10, 32, 16, do_prob=0.1)
    x = torch.randn(2, 5, 10)
    output = mlp(x)
    assert output.shape == (2, 5, 16)
    print("✓ MLP 模块正常")
    
    # 测试MLPEncoder
    batch_size, num_atoms, timesteps, dims = 2, 3, 10, 6
    encoder = MLPEncoder(
        n_in=timesteps * dims,
        n_hid=32,
        n_out=4,  # 4种边类型
        do_prob=0.1,
        factor=True
    )
    
    # 创建输入数据
    inputs = torch.randn(batch_size, num_atoms, timesteps, dims)
    
    # 创建关系矩阵
    num_edges = num_atoms * (num_atoms - 1)
    rel_rec = torch.zeros(num_edges, num_atoms)
    rel_send = torch.zeros(num_edges, num_atoms)
    
    idx = 0
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                rel_rec[idx, j] = 1
                rel_send[idx, i] = 1
                idx += 1
    
    # 前向传播
    output = encoder(inputs, rel_rec, rel_send)
    expected_shape = (batch_size, num_edges, 4)
    assert output.shape == expected_shape
    print("✓ MLPEncoder 模块正常")
    
    # 测试MLPDecoder
    decoder = MLPDecoder(
        n_in_node=dims,
        edge_types=4,
        msg_hid=32,
        msg_out=32,
        n_hid=32,
        do_prob=0.1,
        skip_first=False
    )
    
    # 创建边类型数据
    rel_type = torch.randn(batch_size, num_edges, 4)
    rel_type = torch.softmax(rel_type, dim=-1)
    
    # 前向传播
    pred_steps = 1
    output = decoder(inputs, rel_type, rel_rec, rel_send, pred_steps)
    expected_shape = (batch_size, num_atoms, timesteps - 1, dims)
    assert output.shape == expected_shape
    print("✓ MLPDecoder 模块正常")


def create_visualization():
    """创建可视化演示"""
    print("\n创建可视化演示...")
    
    # 创建示例边概率矩阵
    num_residues = 5
    probs = np.random.rand(num_residues * (num_residues - 1) // 2)
    
    # 创建邻接矩阵
    adj_matrix = np.zeros((num_residues, num_residues))
    idx = 0
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            adj_matrix[i, j] = probs[idx]
            adj_matrix[j, i] = probs[idx]
            idx += 1
    
    # 创建热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Edge Probability')
    plt.title('示例：学习到的残基间相互作用')
    plt.xlabel('残基索引')
    plt.ylabel('残基索引')
    
    # 保存图像
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/demo_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存可视化图像: logs/demo_heatmap.png")


def main():
    """主演示函数"""
    print("=" * 60)
    print("NRI-MD Python 3.10+ 兼容性演示")
    print("=" * 60)
    
    # 显示环境信息
    print(f"Python 版本: {torch.__version__}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    try:
        # 1. 创建示例数据
        features, edges = create_sample_data()
        
        # 2. 测试工具函数
        test_utility_functions()
        
        # 3. 测试神经网络模块
        test_neural_modules()
        
        # 4. 创建可视化
        create_visualization()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！NRI-MD已成功适配Python 3.10+")
        print("=" * 60)
        
        print("\n下一步操作:")
        print("1. 将您的PDB轨迹文件放入 data/pdb/ 目录")
        print("2. 运行数据转换: nri-md-convert --data-dir data/pdb --output-dir data")
        print("3. 训练模型: nri-md-train --num-residues <N> --epochs 100")
        print("4. 可视化结果: nri-md-visualize --filename logs/out_probs_train.npy")
        print("5. 分析路径: nri-md-path --filename logs/out_probs_train.npy --source-node <N>")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()