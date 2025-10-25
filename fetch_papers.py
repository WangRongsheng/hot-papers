import json
import requests
from datetime import datetime
from typing import List, Dict, Optional

def fetch_papers_from_api(page_num: int = 0, page_size: int = 3, sort_by: str = 'Hot') -> Optional[Dict]:
    """
    从API获取论文数据
    
    参数:
        page_num (int): 页码，默认0
        page_size (int): 每页数量，默认3
        sort_by (str): 排序方式 (Hot/Comments/Views/Likes/Github)
    
    返回:
        dict: API响应数据，失败时返回None
    """
    api_url = f"https://api.alphaxiv.org/papers/v2/feed?pageNum={page_num}&sortBy={sort_by}&pageSize={page_size}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return None

def parse_papers(json_data: Dict, sort_by: str = 'Hot') -> List[Dict]:
    """
    解析论文数据并支持排序
    
    参数:
        json_data (dict): API响应的JSON数据
        sort_by (str): 排序方式 (Hot/Comments/Views/Likes/Github)
    
    返回:
        list: 解析后的论文列表
    """
    papers = json_data.get('papers', [])
    parsed_papers = []
    
    for paper in papers:
        # 安全获取嵌套字段
        metrics = paper.get('metrics') or {}
        visits_count = metrics.get('visits_count') or {}
        paper_summary = paper.get('paper_summary') or {}
        
        # 提取基础信息
        parsed_paper = {
            'id': paper.get('id'),
            'title': paper.get('title'),
            'abstract': paper.get('abstract'),
            'image_url': paper.get('image_url'),
            'paper_id': paper.get('universal_paper_id'),
            'publication_date': paper.get('first_publication_date'),
            'topics': paper.get('topics', []),
            'organizations': [org.get('name', '') for org in (paper.get('organization_info') or [])],
            'github_url': paper.get('github_url'),
            'github_stars': paper.get('github_stars', 0),
            'metrics': {
                'votes': metrics.get('total_votes', 0),
                'public_votes': metrics.get('public_total_votes', 0),
                'visits': visits_count.get('all', 0),
                'recent_visits': visits_count.get('last_7_days', 0)
            },
            'summary': paper_summary.get('summary', ''),
            'key_insights': paper_summary.get('keyInsights', []),
            'results': paper_summary.get('results', [])
        }
        
        # 转换日期格式
        if parsed_paper['publication_date']:
            try:
                dt = datetime.fromisoformat(parsed_paper['publication_date'].replace('Z', '+00:00'))
                parsed_paper['publication_date'] = dt.strftime('%Y-%m-%d')
            except:
                pass
        
        parsed_papers.append(parsed_paper)
    
    # 根据参数排序
    if sort_by == 'Hot':
        # 热度 = 投票数 + 访问量权重
        parsed_papers.sort(
            key=lambda x: x['metrics']['votes'] + x['metrics']['visits']*0.01,
            reverse=True
        )
    elif sort_by == 'Comments':
        # 当前API不提供评论数，使用投票数代替
        parsed_papers.sort(key=lambda x: x['metrics']['public_votes'], reverse=True)
    elif sort_by == 'Views':
        parsed_papers.sort(key=lambda x: x['metrics']['visits'], reverse=True)
    elif sort_by == 'Likes':
        parsed_papers.sort(key=lambda x: x['metrics']['public_votes'], reverse=True)
    elif sort_by == 'Github':
        parsed_papers.sort(key=lambda x: x['github_stars'], reverse=True)
    
    return parsed_papers

def get_papers(page_num: int = 0, page_size: int = 3, sort_by: str = 'Hot') -> List[Dict]:
    """
    获取并解析论文数据
    
    参数:
        page_num (int): 页码，默认0
        page_size (int): 每页数量，默认3
        sort_by (str): 排序方式 (Hot/Comments/Views/Likes/Github)
    
    返回:
        list: 解析后的论文列表，失败时返回空列表
    """
    # 从API获取数据
    api_data = fetch_papers_from_api(page_num, page_size, sort_by)
    
    if api_data is None:
        return []
    
    # 解析数据
    return parse_papers(api_data, sort_by)

def save_to_js(data: Dict, filename: str = 'day_paper_info.js'):
    """
    将数据保存到JS文件
    
    参数:
        data (dict): 要保存的数据
        filename (str): 文件名，默认为day_paper_info.js
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('const papersData = ')
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write(';')
        print(f"✅ 数据已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")

# 示例使用
if __name__ == "__main__":
    # 配置参数
    page_num = 0
    page_size = 30
    sort_by = 'Hot'  # 可选: Hot/Comments/Views/Likes/Github
    
    # 获取原始数据并直接保存为JS文件
    api_data = fetch_papers_from_api(page_num, page_size, sort_by)
    
    if api_data:
        save_to_js(api_data, 'day_paper_info.js')
    else:
        print("❌ 未能获取数据")