from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from api import model
import numpy as np

# 定义softmax函数的计算
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # 为了稳定性
    return exp_logits / np.sum(exp_logits)

# 定义 DeRa 函数
def DeRa(logits, reward_factor, regularization_factor):
    adjusted_logits = (reward_factor * logits) + (regularization_factor * logits)
    return adjusted_logits

# 定义markdown表格格式化函数
def format_as_markdown(data):
    headers = data[0].keys()  # 假设每一行都有相同的键
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "|---" * len(headers) + "|\n"
    
    for row in data:
        markdown += "| " + " | ".join(str(row[header]) for header in headers) + " |\n"
    
    return markdown

# 偿债能力分析
# 使用CoVe方式搭建workflow
# 定义三个模型的PromptTemplate

#初步信息抽取及分析
template1 = PromptTemplate(
    template='''
    # 角色
    你是一位资深的财务分析师，善于通过企业的主要财务指标表、资产负债表来分析企业的偿债能力，你有非常精确的信息匹配和抽取能力。

    # 目标
    通过提供的主要财务指标表、资产负债表，结合你的金融领域知识，计算用于评估企业偿债能力的财务指标，并与前两年的数据进行对比，从而揭示出企业的风险点所在。

    # 技能
    - 丰富的股票市场知识
    - 熟练的财务分析技能
    - 对财表隐藏风险点的洞察能力
    - 熟练的python数据分析能力

    # 工作流程
    1.接收财表数据和领域知识
    2.根据实际情况，从输入数据中抽取分析企业偿债能力所需的数据信息
    3.整理数据信息抽取结果，输出你从输入中筛选抽取出的用于计算企业偿债能力分析关键数据信息。

    # 限制
    仅输出用于计算企业偿债能力分析关键数据信息，用markdown表格的形式输出。

    # 输入参数
    主要指标表：{input_sheet_1}
    资产负债表：{input_sheet_2}
    金融领域对应计算指标：{input_rag}

    ''',
    input_variables=['input']
)

# 创建输入数据
input_data = template1.format(input_sheet_1='sheet1_data', input_sheet_4='sheet4_data', rag='rag_data')

# 将输入数据传递给模型
first_output_logits = model(input_data)

# 应用 DeRa
adjusted_output_logits = DeRa(first_output_logits, reward_factor=0.7, regularization_factor=0.3)
first_output = softmax(adjusted_output_logits)

# 格式化输出为 Markdown 表格
first_output_markdown = format_as_markdown(first_output)
print(first_output_markdown)

# 根据template1的结果，进行偿债能力分析所需的指标计算

template2= PromptTemplate(
    template='''
    # 角色
    你是一位资深的财务分析师，善于通过企业的主要财务指标表、资产负债表来分析企业的偿债能力。

    # 目标
    通过提供的主要财务指标表、资产负债表，结合你的金融领域知识，计算用于评估企业偿债能力的财务指标，并与前两年的数据进行对比，从而揭示出企业的风险点所在。

    # 技能
    - 丰富的股票市场知识
    - 熟练的财务分析技能
    - 对财表隐藏风险点的洞察能力
    - 熟练的python数据分析能力

    # 工作流程
    1.接收财表数据和领域知识
    2.根据实际情况，逐步规划你的财务分析步骤，每个步骤应该详尽、科学严谨、可操作性强
    3.根据你规划出的财务分析步骤，使用python编程语言进行分析
    4.整理数据分析结果，输出你的财务分析步骤和计算出的用于企业偿债能力分析的财务指标。

    # 限制
    仅输出你的财务分析步骤和计算出的用于企业偿债能力分析的财务指标。

    # 输入参数
    主要指标表：{input_sheet_1}
    资产负债表：{input_sheet_2}
    初步抽取的数据信息：{first_output_markdown}

    ''',
    input_variables=['input']
)

second_output_logits = model(first_output)

# 应用 DeRa
adjusted_second_logits = DeRa(second_output_logits, reward_factor=0.8, regularization_factor=0.2)
second_output = softmax(adjusted_second_logits)

# 输出分析结果
print(second_output)

# 根据template2的结果，进行偿债能力分析

template3 = PromptTemplate(
    template='''
    # 角色
    你是一位资深的财务分析师，善于通过企业的主要财务指标表、资产负债表来分析企业的偿债能力。

    # 目标
    通过提供的主要财务指标表、资产负债表，结合你的金融领域知识，计算用于评估企业偿债能力的财务指标，并与前两年的数据进行对比，从而揭示出企业的风险点所在。

    # 技能
    - 丰富的股票市场知识
    - 熟练的财务分析技能
    - 对财表隐藏风险点的洞察能力
    - 熟练的python数据分析能力

    # 工作流程
    1.接收财务指标计算结果和领域知识
    2.结合计算出的指标和你的知识，揭露企业在偿债能力上可能出现的风险点。

    # 限制
    仅输出该企业的偿债能力分析及该企业在偿债能力上可能出现的风险点

    # 输入参数
    财务指标计算结果：{input_template1_result}
    金融领域对应计算指标：{rag}

    ''',
    input_variables=['input']
)

third_output_logits = model(second_output)

# 应用 DeRa
adjusted_third_logits = DeRa(third_output_logits, reward_factor=0.75, regularization_factor=0.25)
third_output = softmax(adjusted_third_logits)

# 输出所需指标
print(third_output)

# 定义模型调用
def model1(input):
    first_output_logits = model([HumanMessage(content=template1.format(input=input))]).content
    adjusted_output_logits = DeRa(first_output_logits)
    return softmax(adjusted_output_logits)

def model2(input):
    first_output_logits = model([HumanMessage(content=template2.format(input=input))]).content
    adjusted_output_logits = DeRa(first_output_logits)
    return softmax(adjusted_output_logits)

def model3(input):
    first_output_logits = model([HumanMessage(content=template3.format(input=input))]).content
    adjusted_output_logits = DeRa(first_output_logits)
    return softmax(adjusted_output_logits)

# 封装成一个工作流
def calculate_debt_ability(input_data):

    # 第一步：模型1进行信息抽取及计算财务指标
    calculation_result = model1(input_data)
    
    # 第二步：模型2解释计算结果并进行偿债能力分析
    explanation = model2(calculation_result)
    
    # 第三步：模型3进行进一步分析
    calculate = model3(explanation)

    # 返回最终结果
    return {
        "calculation_result": calculation_result,
        "explanation": explanation,
        "calculate": calculate
    }

# 使用工作流
input_data = '主要指标表数据+资产负债表数据'
results = calculate_debt_ability(input_data)
print("信息抽取结果:", results["calculation_result"])
print("指标计算结果:", results["explanation"])
print("偿债能力分析:", results["calculate"])


# def model1(input):
#     return model([HumanMessage(content=template1.format(input=input))]).content

# def model2(input):
#     return model([HumanMessage(content=template1.format(input=input))]).content

# def model3(input):
#     return model([HumanMessage(content=template2.format(input=input))]).content

# # 封装成一个工作流
# def calculate_debt_ability(input_data):

#     # 第一步：模型1进行信息抽取及计算财务指标
#     calculation_result = model1(input_data)
    
#     # 第二步：模型2解释计算结果并进行偿债能力分析
#     explanation = model2(calculation_result)
    
#     # 第三步：模型3解释计算结果并进行偿债能力分析
#     calculate = model3(explanation)

#     # 返回最终结果
#     return {
#         "calculation_result": calculation_result,
#         "explanation": explanation,
#         "calculate": calculate
#     }

# # 使用工作流
# input_data = '主要指标表数据+资产负债表数据'
# results = calculate_debt_ability(input_data)
# print("信息抽取结果:", results["calculation_result"])
# print("指标计算结果:", results["explanation"])
# print("偿债能力分析:", results["calculate"])
