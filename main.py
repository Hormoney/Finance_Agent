import pandas as pd
from models.model_debt import calculate_debt_ability
from models.model_profit import calculate_profit_ability
from models.model_operation import calculate_operation_ability

# 读取三个Excel数据表
main_data = pd.read_excel('path/to/main_data.xlsx')  # 主要指标表
debt_data = pd.read_excel('path/to/debt_data.xlsx')  # 资产负债表
balance_data = pd.read_excel('path/to/balance_data.xlsx')  # 利润表
indicator_data = pd.read_excel('path/to/indicator_data.xlsx')  # 现金流表

# 将数据转换为字符串
debt_input = debt_data.to_string(index=False)
balance_input = balance_data.to_string(index=False)
indicator_input = indicator_data.to_string(index=False)

# 调用偿债能力分析模型
debt_results = calculate_debt_ability(
    input_sheet_1=debt_input,
    input_sheet_2=balance_input,
    input_rag=indicator_input
)

# 调用盈利能力分析模型
profit_results = calculate_profit_ability(
    input_sheet_1=debt_input,
    input_sheet_3=balance_input,
    input_rag=indicator_input
)

# 调用营运能力分析模型
operation_results = calculate_operation_ability(
    input_sheet_1=debt_input,
    input_sheet_4=balance_input,
    input_rag=indicator_input
)

# 输出结果
print("偿债能力结果:", debt_results)
print("盈利能力结果:", profit_results)
print("营运能力结果:", operation_results)
