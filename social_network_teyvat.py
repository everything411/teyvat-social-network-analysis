import json
import os
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import community as community_louvain  # Louvain社区检测算法
import seaborn as sns
from pyvis.network import Network
import pandas as pd
import datetime

matplotlib.rc("font", family="Microsoft YaHei")

def load_text_map(file_path):
    """加载文本映射文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_avatar_data(file_path, text_map):
    """加载角色数据，创建角色ID到角色名的映射"""
    with open(file_path, 'r', encoding='utf-8') as f:
        avatars = json.load(f)
    
    avatar_id_to_name = {}
    for avatar in avatars:
        text_hash = str(avatar.get('nameTextMapHash', ''))
        if text_hash in text_map:
            avatar_name = text_map[text_hash]
            avatar_id_to_name[avatar['id']] = avatar_name
    
    return avatar_id_to_name

def build_mention_graph(fetters_file, text_map, avatar_id_to_name):
    """构建角色提及关系图，处理角色别名"""
    with open(fetters_file, 'r', encoding='utf-8') as f:
        voice_data = json.load(f)
    
    # 角色别名映射表 (可根据需要扩展)
    alias_map = {
        "仆人": "阿蕾奇诺",
        "公子": "达达利亚",
        "散兵": "流浪者",
        "小吉祥草王": "纳西妲"
        # 添加更多别名映射...
    }

    
    # 创建角色名到提及角色的映射
    graph = defaultdict(set)
    edge_data = []  # 存储边的详细信息
    
    # 创建包含别名和本名的查找集合
    all_names = set(avatar_id_to_name.values())
    for alias, real_name in alias_map.items():
        if real_name in avatar_id_to_name.values():
            all_names.add(alias)
    
    for voice_entry in voice_data:
        avatar_id = voice_entry.get('avatarId')
        if not avatar_id or avatar_id not in avatar_id_to_name:
            continue
            
        text_hash = voice_entry.get('voiceTitleTextMapHash')
        if not text_hash:
            continue
            
        # 获取语音标题文本
        voice_title = text_map.get(str(text_hash), '')
        if not voice_title:
            continue
            
        # 获取说话者名称(使用本名)
        speaker_name = avatar_id_to_name[avatar_id]

        # 跳过旅行者
        if speaker_name == "旅行者":
            continue
        
        # 检查语音标题中提及的角色
        for name in all_names:
            # 跳过自己提及自己
            if name == speaker_name:
                continue
                
            # 检查角色名或别名是否出现在语音标题中
            if name in voice_title:
                # 如果匹配到的是别名，转换为正式名
                mentioned_name = alias_map.get(name, name)
                
                # 确保提及的角色在正式角色列表中
                if mentioned_name in avatar_id_to_name.values():
                    graph[speaker_name].add(mentioned_name)
                    edge_data.append({
                        'source': speaker_name,
                        'target': mentioned_name,
                        'voice_title': voice_title,
                        'avatar_id': avatar_id,
                        'matched_name': name  # 记录实际匹配到的名字(可能是别名)
                    })
    
    return graph, edge_data

# 在 analyze_social_network 函数末尾添加以下代码（在 return 之前）

def generate_html_report(output_dir, analysis_results):
    """生成综合性HTML报告"""
    html_path = os.path.join(output_dir, "index.html")
    
    # 读取生成的各类文件
    try:
        with open(os.path.join(output_dir, "degree_distribution.svg"), 'r', encoding='utf-8') as f:
            degree_dist_svg = f.read()
    except:
        degree_dist_svg = "<p>度分布图未生成</p>"
    
    try:
        with open(os.path.join(output_dir, "community_structure.svg"), 'r', encoding='utf-8') as f:
            community_svg = f.read()
    except:
        community_svg = "<p>社区结构图未生成</p>"

    # 读取社区发现信息
    community_info = ""
    community_file = os.path.join(output_dir, "communities.txt")
    if os.path.exists(community_file):
        with open(community_file, 'r', encoding='utf-8') as f:
            community_info = "<pre>" + f.read() + "</pre>"
    else:
        community_info = "<p>社区发现未生成</p>"
    
    try:
        with open(os.path.join(output_dir, "core_periphery.svg"), 'r', encoding='utf-8') as f:
            core_periphery_svg = f.read()
    except:
        core_periphery_svg = "<p>核心-边缘结构图未生成</p>"
    
    # 读取直径路径信息
    diameter_info = ""
    diameter_path_file = os.path.join(output_dir, "diameter_paths.txt")
    if os.path.exists(diameter_path_file):
        with open(diameter_path_file, 'r', encoding='utf-8') as f:
            diameter_info = "<pre>" + f.read() + "</pre>"
    else:
        diameter_path_file = os.path.join(output_dir, "diameter_paths_undirected.txt")
        if os.path.exists(diameter_path_file):
            with open(diameter_path_file, 'r', encoding='utf-8') as f:
                diameter_info = "<pre>" + f.read() + "</pre>"
        else:
            diameter_info = "<p>直径路径信息未生成</p>"
    
    # 读取互惠关系信息
    reciprocal_info = ""
    reciprocal_file = os.path.join(output_dir, "reciprocal_pairs.txt")
    if os.path.exists(reciprocal_file):
        with open(reciprocal_file, 'r', encoding='utf-8') as f:
            reciprocal_info = "<pre>" + f.read() + "</pre>"
    
    # 创建HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>提瓦特角色关系网络分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #1a5276;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .chart {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow: scroll;
            height: 200px;
        }}
        .nav {{
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .nav a {{
            margin: 0 10px;
            padding: 8px 15px;
            background: #2980b9;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        .nav a:hover {{
            background: #3498db;
        }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">提瓦特角色关系网络分析报告</h1>
    
    <div class="nav">
        <a href="#basic">基本指标</a>
        <a href="#centrality">中心性分析</a>
        <a href="#community">社区结构</a>
        <a href="#core">核心边缘</a>
        <a href="#diameter">直径路径</a>
        <a href="#interactive">交互图</a>
    </div>
    
    <div class="section" id="basic">
        <h2>基本网络指标</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>节点数(角色数)</td><td>{analysis_results['n_nodes']}</td></tr>
            <tr><td>边数(评价关系)</td><td>{analysis_results['n_edges']}</td></tr>
            <tr><td>网络密度</td><td>{analysis_results['density']:.4f}</td></tr>
            <tr><td>最大k-core值</td><td>{analysis_results['max_k']}</td></tr>
            <tr><td>核心节点数</td><td>{len(analysis_results['core_nodes'])}</td></tr>
            <tr><td>社区数量</td><td>{len(analysis_results['communities'])}</td></tr>
        </table>
        
        <h3>度分布</h3>
        <div class="grid">
            <div class="chart">
                {degree_dist_svg}
            </div>
        </div>
    </div>
    
    <div class="section" id="centrality">
        <h2>中心性分析</h2>
        
        <h3>最具影响力角色(中介中心性Top5)</h3>
        <table>
            <tr><th>排名</th><th>角色</th><th>中介中心性</th></tr>
            {"".join([f"<tr><td>{i+1}</td><td>{char}</td><td>{cent:.4f}</td></tr>" 
              for i, (char, cent) in enumerate(analysis_results['top_betweenness'][:5])])}
        </table>
        
        <h3>最受欢迎角色(入度中心性Top5)</h3>
        <table>
            <tr><th>排名</th><th>角色</th><th>入度中心性</th></tr>
            {"".join([f"<tr><td>{i+1}</td><td>{char}</td><td>{cent:.4f}</td></tr>" 
              for i, (char, cent) in enumerate(analysis_results['top_in_centrality'][:5])])}
        </table>
        
        <h3>最活跃角色(出度中心性Top5)</h3>
        <table>
            <tr><th>排名</th><th>角色</th><th>出度中心性</th></tr>
            {"".join([f"<tr><td>{i+1}</td><td>{char}</td><td>{cent:.4f}</td></tr>" 
              for i, (char, cent) in enumerate(analysis_results['top_out_centrality'][:5])])}
        </table>
    </div>
    
    <div class="section" id="community">
        <h2>社区结构</h2>
        <div class="chart">
            {community_svg}
        </div>

        <h3>社区发现</h3>
        {community_info}
        
        <h3>最大社区成员</h3>
        <p>{", ".join(analysis_results['communities'][0][1][:20])}...</p>
    </div>
    
    <div class="section" id="core">
        <h2>核心-边缘结构</h2>
        <div class="chart">
            {core_periphery_svg}
        </div>
        
        <h3>核心节点</h3>
        <p>{", ".join(analysis_results['core_nodes'])}</p>
    </div>
    
    <div class="section" id="diameter">
        <h2>直径路径分析</h2>
        {diameter_info}
    </div>
    
    <div class="section" id="reciprocal">
        <h2>互惠关系分析</h2>
        {reciprocal_info if reciprocal_info else "<p>无互惠关系</p>"}
    </div>
    
    <div class="section" id="interactive">
        <h2>交互式网络图</h2>
        <iframe src="interactive_network.html" width="100%" height="800px" style="border:none;"></iframe>
    </div>
    
    <footer style="text-align: center; margin-top: 50px; color: #777;">
        <p>报告生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
</body>
</html>
    """
    
    # 写入HTML文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"综合性HTML报告已生成: {html_path}")

def analyze_social_network(G, output_dir="social_network_analysis"):
    """全面分析社交网络"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 基本网络指标
    print("\n===== 基本网络指标 =====")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    is_directed = nx.is_directed(G)
    
    print(f"节点数: {n_nodes}")
    print(f"边数: {n_edges}")
    print(f"网络密度: {density:.4f}")
    print(f"是否为有向图: {is_directed}")
    
    # 2. 度分布分析
    print("\n===== 度分布分析 =====")
    if is_directed:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        avg_in_degree = sum(in_degrees.values()) / n_nodes
        avg_out_degree = sum(out_degrees.values()) / n_nodes
        
        print(f"平均入度: {avg_in_degree:.2f}")
        print(f"平均出度: {avg_out_degree:.2f}")
        
        # 保存度分布数据
        degree_df = pd.DataFrame({
            'character': list(in_degrees.keys()),
            'in_degree': list(in_degrees.values()),
            'out_degree': list(out_degrees.values())
        })
        degree_df.to_csv(os.path.join(output_dir, "degree_distribution.csv"), index=False)
        
        # 度分布可视化
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(list(in_degrees.values()), bins=20, kde=True, color='skyblue')
        plt.title('入度分布', fontsize=14, fontname='SimHei')
        plt.xlabel('入度')
        plt.ylabel('频数')
        
        plt.subplot(1, 2, 2)
        sns.histplot(list(out_degrees.values()), bins=20, kde=True, color='salmon')
        plt.title('出度分布', fontsize=14, fontname='SimHei')
        plt.xlabel('出度')
        plt.ylabel('频数')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "degree_distribution.svg"), dpi=300)
        plt.close()
        
    else:
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / n_nodes
        print(f"平均度: {avg_degree:.2f}")
    
    # 3. 中心性分析
    print("\n===== 中心性分析 =====")
    
    # 度中心性
    if is_directed:
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)
        
        # 找出最中心节点
        top_in_centrality = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_out_centrality = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n入度中心性Top 10:")
        for char, cent in top_in_centrality:
            print(f"  {char}: {cent:.4f}")
        
        print("\n出度中心性Top 10:")
        for char, cent in top_out_centrality:
            print(f"  {char}: {cent:.4f}")
        
        # 保存中心性数据
        centrality_df = pd.DataFrame({
            'character': list(in_degree_centrality.keys()),
            'in_degree_centrality': list(in_degree_centrality.values()),
            'out_degree_centrality': list(out_degree_centrality.values())
        })
        centrality_df.to_csv(os.path.join(output_dir, "degree_centrality.csv"), index=False)
    
    # 接近中心性（无向图）
    if nx.is_strongly_connected(G) or nx.is_weakly_connected(G):
        if is_directed:
            # 对于有向图，我们使用强连通分量或弱连通分量
            if nx.is_strongly_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
            else:
                closeness_centrality = nx.closeness_centrality(G.to_undirected())
        else:
            closeness_centrality = nx.closeness_centrality(G)
        
        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n接近中心性Top 10:")
        for char, cent in top_closeness:
            print(f"  {char}: {cent:.4f}")
        
        centrality_df['closeness_centrality'] = centrality_df['character'].map(
            lambda x: closeness_centrality.get(x, np.nan))
        centrality_df.to_csv(os.path.join(output_dir, "centrality_metrics.csv"), index=False)
    
    # 中介中心性
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n中介中心性Top 10:")
    for char, cent in top_betweenness:
        print(f"  {char}: {cent:.4f}")
    
    centrality_df['betweenness_centrality'] = centrality_df['character'].map(
        lambda x: betweenness_centrality.get(x, np.nan))
    centrality_df.to_csv(os.path.join(output_dir, "centrality_metrics.csv"), index=False)
    
    # 特征向量中心性
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n特征向量中心性Top 10:")
        for char, cent in top_eigenvector:
            print(f"  {char}: {cent:.4f}")
        
        centrality_df['eigenvector_centrality'] = centrality_df['character'].map(
            lambda x: eigenvector_centrality.get(x, np.nan))
        centrality_df.to_csv(os.path.join(output_dir, "centrality_metrics.csv"), index=False)
    except nx.PowerIterationFailedConvergence:
        print("特征向量中心性计算未收敛")
    
    # 4. 社区检测
    print("\n===== 社区检测 =====")
    # 转换为无向图进行社区检测
    undirected_G = G.to_undirected()
    
    # 使用Louvain算法检测社区
    partition = community_louvain.best_partition(undirected_G)
    
    # 统计社区信息
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    print(f"检测到 {len(communities)} 个社区")
    
    # 按社区大小排序
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 保存社区信息
    with open(os.path.join(output_dir, "communities.txt"), 'w', encoding='utf-8') as f:
        f.write(f"共检测到 {len(communities)} 个社区\n\n")
        for comm_id, members in sorted_communities:
            f.write(f"社区 {comm_id} (成员数: {len(members)}):\n")
            f.write(", ".join(members) + "\n\n")
    
    # 可视化社区结构
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(undirected_G, seed=42)
    
    # 绘制节点，按社区着色
    cmap = plt.cm.tab20
    for comm_id in communities:
        nx.draw_networkx_nodes(
            undirected_G, pos,
            nodelist=communities[comm_id],
            node_color=[cmap(comm_id % 20)],
            node_size=100,
            alpha=0.8
        )
    
    # 绘制边
    nx.draw_networkx_edges(undirected_G, pos, alpha=0.2)
    
    # 标注重要节点
    top_nodes = [node for node, _ in top_betweenness[:5]]
    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(undirected_G, pos, labels, font_size=10, font_family='SimHei')
    
    plt.title("提瓦特角色评价关系社区结构", fontsize=16, fontname='SimHei')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "community_structure.svg"), dpi=300)
    plt.close()
    
    # 5. 连通性分析
    print("\n===== 连通性分析 =====")
    if is_directed:
        # 强连通分量
        scc = list(nx.strongly_connected_components(G))
        scc_sizes = sorted([len(comp) for comp in scc], reverse=True)
        largest_scc = max(scc, key=len)
        
        print(f"强连通分量数量: {len(scc)}")
        print(f"最大强连通分量大小: {len(largest_scc)}")
        print(f"强连通分量大小分布: {scc_sizes}")
        print("强连通分量：", scc)
        
        # 弱连通分量
        wcc = list(nx.weakly_connected_components(G))
        wcc_sizes = sorted([len(comp) for comp in wcc], reverse=True)
        largest_wcc = max(wcc, key=len)
        
        print(f"\n弱连通分量数量: {len(wcc)}")
        print(f"最大弱连通分量大小: {len(largest_wcc)}")
        print(f"弱连通分量大小分布: {wcc_sizes}")
        
        # 绘制连通分量大小分布
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(scc_sizes)+1), scc_sizes, color='skyblue')
        plt.title('强连通分量大小分布', fontsize=14, fontname='SimHei')
        plt.xlabel('分量排名')
        plt.ylabel('大小')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(wcc_sizes)+1), wcc_sizes, color='salmon')
        plt.title('弱连通分量大小分布', fontsize=14, fontname='SimHei')
        plt.xlabel('分量排名')
        plt.ylabel('大小')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "connected_components.svg"), dpi=300)
        plt.close()
    
    # 6. 路径分析
    print("\n===== 路径分析 =====")
    # 平均最短路径长度
    if nx.is_strongly_connected(G) or nx.is_weakly_connected(G):
        if is_directed and nx.is_strongly_connected(G):
            avg_shortest_path = nx.average_shortest_path_length(G)
            print(f"平均最短路径长度: {avg_shortest_path:.2f}")
        else:
            # 对于非强连通图，使用弱连通分量的平均
            avg_shortest_path = nx.average_shortest_path_length(G.to_undirected())
            print(f"平均最短路径长度(无向): {avg_shortest_path:.2f}")
    
    # 直径（最长最短路径）分析
    if nx.is_strongly_connected(G):
        # 计算所有节点对的最短路径长度
        all_pairs_shortest = dict(nx.all_pairs_shortest_path_length(G))
        
        # 找出所有达到直径长度的路径
        diameter = nx.diameter(G)
        diameter_paths = []
        
        # 遍历所有节点对
        for source in G.nodes():
            for target, length in all_pairs_shortest[source].items():
                if length == diameter:
                    # 获取具体路径
                    paths = list(nx.all_shortest_paths(G, source, target))
                    for path in paths:
                        diameter_paths.append(path)
        
        print(f"网络直径: {diameter}")
        print(f"达到直径长度的路径数量: {len(diameter_paths)}")
        
        # 打印前5条路径（避免输出过多）
        print("\n示例路径（最多显示5条）:")
        for i, path in enumerate(diameter_paths[:5]):
            print(f"路径 {i+1}: {' → '.join(path)}")
            
        # 保存所有直径路径到文件
        with open(os.path.join(output_dir, "diameter_paths.txt"), 'w', encoding='utf-8') as f:
            f.write(f"网络直径: {diameter}\n")
            f.write(f"达到直径长度的路径总数: {len(diameter_paths)}\n\n")
            for i, path in enumerate(diameter_paths):
                f.write(f"路径 {i+1}: {' → '.join(path)}\n")
    
    elif nx.is_weakly_connected(G):
        # 对于弱连通图，转换为无向图处理
        undirected_G = G.to_undirected()
        diameter = nx.diameter(undirected_G)
        
        # 计算所有节点对的最短路径长度
        all_pairs_shortest = dict(nx.all_pairs_shortest_path_length(undirected_G))
        
        diameter_paths = []
        for source in undirected_G.nodes():
            for target, length in all_pairs_shortest[source].items():
                if length == diameter:
                    paths = list(nx.all_shortest_paths(undirected_G, source, target))
                    for path in paths:
                        diameter_paths.append(path)
        
        print(f"网络直径(无向): {diameter}")
        print(f"达到直径长度的路径数量: {len(diameter_paths)}")
        
        print("\n示例路径（最多显示5条）:")
        for i, path in enumerate(diameter_paths[:5]):
            print(f"路径 {i+1}: {' → '.join(path)}")
            
        with open(os.path.join(output_dir, "diameter_paths_undirected.txt"), 'w', encoding='utf-8') as f:
            f.write(f"网络直径(无向): {diameter}\n")
            f.write(f"达到直径长度的路径总数: {len(diameter_paths)}\n\n")
            for i, path in enumerate(diameter_paths):
                f.write(f"路径 {i+1}: {' → '.join(path)}\n")
    else:
        print("网络不连通，无法计算直径")
    
    # 7. 互惠性分析
    print("\n===== 互惠性分析 =====")
    if is_directed:
        reciprocity = nx.reciprocity(G)
        print(f"互惠边比例: {reciprocity:.4f}")
        
        # 找出互惠关系对
        reciprocal_pairs = []
        for u, v in G.edges():
            if G.has_edge(v, u):
                reciprocal_pairs.append((u, v))
        
        print(f"互惠关系对数量: {len(reciprocal_pairs)//2}")
        
        # 保存互惠关系
        with open(os.path.join(output_dir, "reciprocal_pairs.txt"), 'w', encoding='utf-8') as f:
            f.write(f"互惠关系对 (共 {len(reciprocal_pairs)//2} 对):\n\n")
            for i in range(0, len(reciprocal_pairs), 2):
                u, v = reciprocal_pairs[i]
                f.write(f"{u} ↔ {v}\n")
    
    # 8. 核心-边缘结构分析 (修复版本)
    print("\n===== 核心-边缘结构分析 =====")
    # 使用k-core分解
    if is_directed:
        # 对于有向图，使用k-shell分解
        k_shell = nx.core_number(G)
    else:
        k_shell = nx.core_number(undirected_G)
    
    # 找出核心节点
    max_k = max(k_shell.values()) if k_shell else 0
    core_nodes = [node for node, k in k_shell.items() if k == max_k]
    
    print(f"最大k-core值: {max_k}")
    print(f"核心节点数量: {len(core_nodes)}")
    print("核心节点:", ", ".join(core_nodes))
    
    # 可视化核心-边缘结构 (修复后的代码)
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(undirected_G, seed=42)
    
    # 绘制节点，根据k-core值着色
    nodes = nx.draw_networkx_nodes(
        undirected_G, pos,
        node_color=[k_shell[node] for node in undirected_G.nodes()],
        cmap=plt.cm.viridis,
        node_size=100,
        alpha=0.8
    )
    
    # 绘制边
    nx.draw_networkx_edges(undirected_G, pos, alpha=0.2)
    
    # 添加颜色条 (修复部分)
    plt.colorbar(nodes, label='k-core值')
    
    # 标注核心节点
    labels = {node: node for node in core_nodes[:10]}  # 只标注前10个核心节点
    nx.draw_networkx_labels(undirected_G, pos, labels, font_size=10, font_family='SimHei')
    
    plt.title("提瓦特角色评价关系核心-边缘结构", fontsize=16, fontname='SimHei')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "core_periphery.svg"), dpi=300)
    plt.close()
    
    # 9. 可视化交互式图
    print("\n生成交互式可视化...")
    nt = Network(
        height='800px', 
        width='100%', 
        directed=is_directed,
        bgcolor='#222222', 
        font_color='white',
        notebook=False
    )
    
    # 添加节点，根据社区着色
    for node in G.nodes():
        comm_id = partition.get(node, 0)
        nt.add_node(
            node, 
            label=node, 
            title=f"{node}\n社区: {comm_id}\n中介中心性: {betweenness_centrality.get(node, 0):.3f}",
            group=comm_id
        )
    
    # 添加边
    for u, v in G.edges():
        nt.add_edge(u, v)
    
    # 配置物理布局
    nt.barnes_hut(
        gravity=-80000,
        central_gravity=0.3,
        spring_length=250,
        spring_strength=0.001,
        damping=0.09,
        overlap=0
    )
    
    # 保存交互式HTML文件
    nt.save_graph(os.path.join(output_dir, "interactive_network.html"))
    print("交互式可视化已保存")

    # 在 analyze_social_network 函数的 return 语句前调用
    generate_html_report(output_dir, {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'communities': sorted_communities,
        'core_nodes': core_nodes,
        'max_k': max_k,
        'top_in_centrality': top_in_centrality,
        'top_out_centrality': top_out_centrality,
        'top_betweenness': top_betweenness
    })
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'communities': sorted_communities,
        'core_nodes': core_nodes,
        'top_in_centrality': top_in_centrality,
        'top_out_centrality': top_out_centrality,
        'top_betweenness': top_betweenness
    }

def main():
    # 文件路径 - 根据实际情况修改
    base_path = '.'  # 包含数据文件的目录
    text_map_file = os.path.join(base_path, 'TextMapCHS.json')
    fetters_file = os.path.join(base_path, 'FettersExcelConfigData.json')
    avatar_file = os.path.join(base_path, 'AvatarExcelConfigData.json')
    
    # 加载数据
    print("Loading text map...")
    text_map = load_text_map(text_map_file)
    print(f"Loaded {len(text_map)} text mappings")
    
    print("Loading avatar data...")
    avatar_id_to_name = load_avatar_data(avatar_file, text_map)
    print(f"Loaded {len(avatar_id_to_name)} avatars")
    
    print("Building mention graph...")
    graph, edge_data = build_mention_graph(fetters_file, text_map, avatar_id_to_name)
    print(f"Graph built with {len(graph)} nodes and {sum(len(v) for v in graph.values())} edges")
    
    # 创建NetworkX图
    G = nx.DiGraph()
    for speaker, mentioned in graph.items():
        G.add_node(speaker)
        for char in mentioned:
            G.add_node(char)
            G.add_edge(speaker, char)
    
    # 进行社交网络分析
    print("\nStarting social network analysis...")
    analysis_results = analyze_social_network(G)
    
    # 打印关键结果
    print("\n===== 关键分析结果总结 =====")
    print(f"网络规模: {analysis_results['n_nodes']} 个角色, {analysis_results['n_edges']} 条评价关系")
    print(f"网络密度: {analysis_results['density']:.4f}")
    print(f"检测到 {len(analysis_results['communities'])} 个社区")
    print(f"核心节点: {', '.join(analysis_results['core_nodes'])}")
    
    print("\n最具影响力的角色(高中介中心性):")
    for char, cent in analysis_results['top_betweenness'][:5]:
        print(f"  {char}: {cent:.4f}")
    
    print("\n分析完成！结果已保存至 social_network_analysis 目录")

if __name__ == "__main__":
    main()