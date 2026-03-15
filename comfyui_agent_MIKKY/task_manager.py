"""
任务队列与并发管理 - 控制并发数、队列大小、任务调度
"""

import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Optional, Any
from datetime import datetime

from .config import DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_QUEUE_SIZE


class TaskManager:
    """任务队列管理器 - 支持并发控制和渐进式回调"""
    
    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    ):
        """
        初始化任务管理器
        
        Args:
            max_concurrent: 最大并发任务数
            max_queue_size: 最大队列大小
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        # 线程安全的状态管理
        self.lock = threading.RLock()
        self.running_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # 结果存储（按任务 ID 索引）
        self.results = {}
        
        print(f"[TaskManager] 初始化: 最大并发={max_concurrent}, 队列大小={max_queue_size}")
    
    def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
        task_executor: Callable[[Dict], Dict],
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> List[Dict]:
        """
        批量提交任务并并发执行
        
        Args:
            tasks: 任务列表，每个任务是一个字典，必须包含 'task_id' 字段
            task_executor: 任务执行函数，接收任务字典，返回结果字典
            progress_callback: 进度回调函数(completed, total, result)
            
        Returns:
            结果列表，按任务提交顺序排列
        """
        total = len(tasks)
        
        if total == 0:
            print("[TaskManager] ⚠️ 任务列表为空")
            return []
        
        if total > self.max_queue_size:
            print(f"[TaskManager] ⚠️ 任务数 ({total}) 超过队列大小 ({self.max_queue_size})，将分批处理")
        
        print(f"[TaskManager] 开始批量执行: 共 {total} 个任务，最大并发 {self.max_concurrent}")
        
        # 重置状态
        with self.lock:
            self.running_tasks = 0
            self.completed_tasks = 0
            self.failed_tasks = 0
            self.results = {}
        
        # 包装任务执行函数，添加错误处理和进度跟踪
        def wrapped_executor(task: Dict) -> Dict:
            task_id = task.get('task_id', 'unknown')
            task_name = task.get('name', f'Task {task_id}')
            
            # 更新运行状态
            with self.lock:
                self.running_tasks += 1
            
            print(f"[TaskManager] 🔄 开始任务 {task_id}: {task_name}")
            
            try:
                # 执行任务
                result = task_executor(task)
                
                # 标记成功
                result['task_id'] = task_id
                result['task_name'] = task_name
                result['success'] = result.get('success', True)
                
                with self.lock:
                    self.completed_tasks += 1
                    if not result['success']:
                        self.failed_tasks += 1
                
                status = "✅" if result['success'] else "❌"
                print(f"[TaskManager] {status} 任务 {task_id} 完成: {task_name}")
                
                # 调用进度回调
                if progress_callback:
                    with self.lock:
                        completed = self.completed_tasks
                    progress_callback(completed, total, result)
                
                return result
                
            except Exception as e:
                # 错误处理
                error_msg = f"任务执行异常: {str(e)}"
                print(f"[TaskManager] ❌ 任务 {task_id} 失败: {error_msg}")
                
                with self.lock:
                    self.completed_tasks += 1
                    self.failed_tasks += 1
                
                result = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'success': False,
                    'error': error_msg
                }
                
                # 调用进度回调
                if progress_callback:
                    with self.lock:
                        completed = self.completed_tasks
                    progress_callback(completed, total, result)
                
                return result
                
            finally:
                # 更新运行状态
                with self.lock:
                    self.running_tasks -= 1
        
        # 使用线程池并发执行
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(wrapped_executor, task): task
                for task in tasks
            }
            
            # 等待所有任务完成（按完成顺序处理）
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get('task_id', 'unknown')
                
                try:
                    result = future.result()
                    results_dict[task_id] = result
                except Exception as e:
                    # 理论上不应该到达这里（已在 wrapped_executor 中处理）
                    print(f"[TaskManager] ❌ 任务 {task_id} 异常: {str(e)}")
                    results_dict[task_id] = {
                        'task_id': task_id,
                        'success': False,
                        'error': str(e)
                    }
        
        # 按原始任务顺序排列结果
        results = []
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            result = results_dict.get(task_id, {
                'task_id': task_id,
                'success': False,
                'error': '任务未执行或丢失'
            })
            results.append(result)
        
        # 输出统计
        with self.lock:
            completed = self.completed_tasks
            failed = self.failed_tasks
            success = completed - failed
        
        print(f"[TaskManager] ✅ 批量执行完成: 总计 {total}, 成功 {success}, 失败 {failed}")
        
        return results
    
    def get_status(self) -> Dict:
        """
        获取当前状态
        
        Returns:
            状态字典
        """
        with self.lock:
            return {
                'running_tasks': self.running_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_tasks': self.completed_tasks - self.failed_tasks,
                'max_concurrent': self.max_concurrent,
                'max_queue_size': self.max_queue_size
            }


class Task:
    """任务对象（便捷类）"""
    
    def __init__(
        self,
        task_id: Any,
        name: str,
        data: Optional[Dict] = None
    ):
        """
        创建任务对象
        
        Args:
            task_id: 任务 ID（唯一标识）
            name: 任务名称
            data: 任务数据（任意键值对）
        """
        self.task_id = task_id
        self.name = name
        self.data = data or {}
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            **self.data
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Task':
        """从字典创建"""
        task_id = d.get('task_id')
        name = d.get('name', f'Task {task_id}')
        data = {k: v for k, v in d.items() if k not in ['task_id', 'name', 'created_at']}
        return cls(task_id, name, data)


# 便捷函数
def execute_batch(
    tasks: List[Dict],
    task_executor: Callable[[Dict], Dict],
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    批量执行任务（便捷函数）
    
    Args:
        tasks: 任务列表
        task_executor: 任务执行函数
        max_concurrent: 最大并发数
        progress_callback: 进度回调
        
    Returns:
        结果列表
    """
    manager = TaskManager(max_concurrent=max_concurrent)
    return manager.submit_batch(tasks, task_executor, progress_callback)
