import json
import requests
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 从环境变量获取API密钥列表
def get_api_keys() -> List[str]:
    """
    从环境变量获取API密钥列表
    支持两种格式：
    1. OPENAI_API_KEYS: JSON数组格式 ["key1", "key2"]
    2. OPENAI_API_KEY_1, OPENAI_API_KEY_2...: 独立环境变量
    """
    api_keys = []
    
    # 方法1: 从JSON数组格式的环境变量读取
    keys_json = os.getenv('OPENAI_API_KEYS')
    if keys_json:
        try:
            api_keys = json.loads(keys_json)
            print(f"✅ 从 OPENAI_API_KEYS 读取到 {len(api_keys)} 个API密钥")
            return api_keys
        except json.JSONDecodeError:
            print("⚠️  OPENAI_API_KEYS 格式错误，尝试其他方式")
    
    # 方法2: 从独立环境变量读取 (OPENAI_API_KEY_1, OPENAI_API_KEY_2, ...)
    index = 1
    while True:
        key = os.getenv(f'OPENAI_API_KEY_{index}')
        if key:
            api_keys.append(key)
            index += 1
        else:
            break
    
    if api_keys:
        print(f"✅ 从 OPENAI_API_KEY_* 读取到 {len(api_keys)} 个API密钥")
        return api_keys
    
    # 方法3: 从单个环境变量读取
    single_key = os.getenv('OPENAI_API_KEY')
    if single_key:
        print(f"✅ 从 OPENAI_API_KEY 读取到 1 个API密钥")
        return [single_key]
    
    print("❌ 未找到任何API密钥，请设置环境变量")
    return []

# 获取API密钥
API_KEYS = get_api_keys()
API_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.ai-gaochao.cn/v1')

# 线程锁用于进度显示
progress_lock = Lock()
progress_counter = {'current': 0, 'total': 0}

def get_output(abstract: str, api_key: str) -> Optional[str]:
    """
    调用模型翻译
    
    参数:
        abstract (str): 要翻译的英文摘要
        api_key (str): OpenAI API密钥
    
    返回:
        str: 翻译后的中文文本，失败时返回None
    """
    try:
        # 每个线程创建独立的client
        client = OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL
        )
        
        prompt = f"请你将下列内容翻译成流畅的中文，不要任何解释，直接翻译。 \n 英文内容为：{abstract}"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            timeout=30
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  翻译失败: {e}")
        return None

def translate_abstract_with_retry(abstract: str, max_retries: int = 2) -> str:
    """
    使用多个API密钥重试翻译
    
    参数:
        abstract (str): 要翻译的英文摘要
        max_retries (int): 每个API key的最大重试次数
    
    返回:
        str: 翻译后的中文文本，所有尝试失败时返回原文
    """
    if not abstract or abstract.strip() == "":
        return abstract
    
    if not API_KEYS:
        print("  ⚠️  没有可用的API密钥，保持英文")
        return abstract
    
    # 遍历所有API密钥
    for idx, api_key in enumerate(API_KEYS):
        # 每个API key尝试多次
        for retry in range(max_retries):
            result = get_output(abstract, api_key)
            
            if result:
                return result
            
            if retry < max_retries - 1:
                wait_time = 1 + retry * 0.5  # 短暂等待
                time.sleep(wait_time)
    
    # 所有API均失败，返回原文
    return abstract

def translate_single_paper(paper: Dict, index: int, total: int) -> Dict:
    """
    翻译单篇论文的摘要（线程工作函数）
    
    参数:
        paper (dict): 论文数据
        index (int): 论文索引
        total (int): 论文总数
    
    返回:
        dict: 翻译后的论文数据
    """
    abstract = paper.get('abstract', '')
    title = paper.get('title', 'Unknown')[:50]
    
    if not abstract:
        with progress_lock:
            progress_counter['current'] += 1
            current = progress_counter['current']
        print(f"[{current}/{total}] 跳过（无摘要）: {title}...")
        return paper
    
    # 开始翻译
    print(f"[线程] 开始翻译: {title}...")
    translated = translate_abstract_with_retry(abstract)
    paper['abstract'] = translated
    
    # 更新进度
    with progress_lock:
        progress_counter['current'] += 1
        current = progress_counter['current']
    
    status = "✅" if translated != abstract else "⚠️"
    print(f"[{current}/{total}] {status} 完成: {title}...")
    
    return paper

def translate_papers_abstracts_multithread(data: Dict, max_workers: int = 32) -> Dict:
    """
    使用多线程翻译所有论文的摘要
    
    参数:
        data (dict): 包含papers的API数据
        max_workers (int): 最大线程数，默认32
    
    返回:
        dict: 翻译后的数据
    """
    papers = data.get('papers', [])
    total = len(papers)
    
    if total == 0:
        print("没有论文需要翻译")
        return data
    
    if not API_KEYS:
        print("⚠️  没有可用的API密钥，跳过翻译")
        return data
    
    # 重置进度计数器
    progress_counter['current'] = 0
    progress_counter['total'] = total
    
    print(f"\n{'='*60}")
    print(f"开始使用 {max_workers} 个线程翻译 {total} 篇论文的摘要...")
    print(f"可用API密钥数量: {len(API_KEYS)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # 使用线程池并发翻译
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有翻译任务
        future_to_index = {
            executor.submit(translate_single_paper, paper, idx, total): idx 
            for idx, paper in enumerate(papers)
        }
        
        # 收集结果
        translated_papers = [None] * total
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                translated_paper = future.result()
                translated_papers[idx] = translated_paper
            except Exception as e:
                print(f"❌ 线程执行出错 (索引 {idx}): {e}")
                # 发生错误时保留原始数据
                translated_papers[idx] = papers[idx]
    
    # 更新数据
    data['papers'] = translated_papers
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 翻译完成！耗时: {elapsed_time:.2f} 秒")
    print(f"{'='*60}\n")
    
    return data

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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='获取并翻译论文数据')
    parser.add_argument('--page-num', type=int, default=0, help='页码，默认0')
    parser.add_argument('--page-size', type=int, default=30, help='每页数量，默认30')
    parser.add_argument('--sort-by', type=str, default='Hot', 
                        choices=['Hot', 'Comments', 'Views', 'Likes', 'Github'],
                        help='排序方式，默认Hot')
    parser.add_argument('--max-workers', type=int, default=32, help='翻译线程数，默认32')
    parser.add_argument('--output', type=str, default='day_paper_info.js', help='输出文件名')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("开始获取论文数据...")
    print(f"配置: page_num={args.page_num}, page_size={args.page_size}, sort_by={args.sort_by}")
    print("=" * 60)
    
    # 获取原始数据
    api_data = fetch_papers_from_api(args.page_num, args.page_size, args.sort_by)
    
    if api_data:
        # 使用多线程翻译所有论文的摘要
        translated_data = translate_papers_abstracts_multithread(api_data, max_workers=args.max_workers)
        
        # 保存为JS文件
        save_to_js(translated_data, args.output)
        
        print("=" * 60)
        print("✅ 所有任务完成！")
        print("=" * 60)
    else:
        print("❌ 未能获取数据")
