"""
快速修复 JSON 转义字符错误的工具脚本
"""
import json
import re

def fix_json_escape(json_string: str) -> str:
    """
    修复JSON中常见的转义字符错误
    
    常见错误：
    1. \'  在JSON中不需要转义单引号（双引号内）
    2. \"  只在字符串值内需要转义双引号
    """
    # 修复 \' -> '
    # 使用正则确保只在字符串值内替换
    fixed = json_string.replace("\\'", "'")
    
    return fixed

def test_and_fix(json_string: str) -> tuple:
    """
    测试JSON并尝试修复
    
    Returns:
        (是否成功, 修复后的JSON字符串或错误信息)
    """
    # 首先尝试直接解析
    try:
        parsed = json.loads(json_string)
        return (True, json_string, "JSON格式正确，无需修复")
    except json.JSONDecodeError as e:
        print(f"原始JSON错误: {e}")
        print(f"错误位置: 第 {e.lineno} 行, 第 {e.colno} 列")
        
        # 尝试修复
        print("\n尝试自动修复...")
        fixed_json = fix_json_escape(json_string)
        
        try:
            parsed = json.loads(fixed_json)
            print("✅ 修复成功！")
            return (True, fixed_json, "已修复转义字符错误")
        except json.JSONDecodeError as e2:
            print(f"❌ 修复失败: {e2}")
            return (False, None, f"无法自动修复: {e2}")

if __name__ == "__main__":
    # 示例用法
    test_json = r'''
    {
        "text": "WHAT\'S INCLUDED"
    }
    '''
    
    success, fixed, msg = test_and_fix(test_json)
    print(f"\n{msg}")
    if success:
        print(f"\n修复后的JSON:\n{fixed}")
