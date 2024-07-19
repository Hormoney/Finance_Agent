from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from api import model

# 营运能力分析
# 使用CoVe方式搭建workflow
# 定义三个模型的PromptTemplate

# 初步信息抽取及分析
template1 = PromptTemplate(
    template='''
    # 角色
    你是一位资深的财务分析师，善于通过企业的主要财务指标表、现金流量表来分析企业的营运能力，你有非常精确的信息匹配和抽取能力。

    # 目标
    通过提供的主要财务指标表、现金流量表，结合你的金融领域知识，计算用于评估企业营运能力的财务指标，并与前两年的数据进行对比，从而揭示出企业的风险点所在。

    # 技能
    - 丰富的股票市场知识
    - 熟练的财务分析技能
    - 对财表隐藏风险点的洞察能力
    - 熟练的python数据分析能力

    # 工作流程
    1.接收财表数据和领域知识
    2.根据实际情况，从输入数据中抽取分析企业营运能力所需的数据信息
    3.整理数据信息抽取结果，输出你从输入中筛选抽取出的用于计算企业营运能力分析关键数据信息。

    # 限制
    仅输出用于计算企业营运能力分析关键数据信息，用markdown表格的形式输出。

    # 输入参数
    主要指标表：{input_sheet_1}
    现金流量表：{input_sheet_4}
    金融领域知识：{rag}

    ''',
    input_variables=['input']
)

# 使用CoVe进行信息抽取验证
# 验证计划制定
template2 = PromptTemplate(
    template='''
    #角色
    你是一名资深财务分析师，专注于从表格数据中抽取有效数据信息和验证财务报告数据的准确性。

    #技能
    根据初步关键数据抽取结果，制定验证计划，列出核实问题。

    #目标
    基于初步关键数据抽取结果{initial_analysis}，制定一个核实计划，列出需要验证的关键问题，确保初步分析的准确性。

    #输入
    初步关键数据抽取结果：{initial_analysis}

    #输出
    输出验证计划，包括需要验证的关键问题列表。

    ''',
    input_variables=['input']
)

# 验证与修正

template3 = PromptTemplate(
    template='''
    #角色
    你是一名资深财务分析师，专注于审计和验证财务报告数据的准确性，以及修改不准确信息，使报告数据准确。

    #技能
    逐一回答核实问题，验证初步关键数据抽取结果，发现并修正错误。

    #目标
    根据核实计划{verification_plan}，逐一回答每个问题，并与初步关键数据抽取结果{initial_analysis}进行比较。指出不一致或错误的地方，并提供修正后的答案。

    #输入
    核实计划：{verification_plan}
    初步关键数据抽取结果：{initial_analysis}

    #输出
    修正后的关键数据。

    ''',
    input_variables=['input']
)

# 根据template3的结果，进行营运能力分析所需的指标计算

template4= PromptTemplate(
    template='''
    # 角色
    你是一位资深的财务分析师，善于通过企业的主要财务指标表、现金流量表来分析企业的营运能力。

    # 目标
    通过提供的主要财务指标表、现金流量表，结合你的金融领域知识，计算用于评估企业营运能力的财务指标，并与前两年的数据进行对比，从而揭示出企业的风险点所在。

    # 技能
    - 丰富的股票市场知识
    - 熟练的财务分析技能
    - 对财表隐藏风险点的洞察能力
    - 熟练的python数据分析能力

    # 工作流程
    1.接收财表数据和领域知识
    2.根据实际情况，逐步规划你的财务分析步骤，每个步骤应该详尽、科学严谨、可操作性强
    3.根据你规划出的财务分析步骤，使用python编程语言进行分析
    4.整理数据分析结果，输出你的财务分析步骤和计算出的用于企业营运能力分析的财务指标。

    # 限制
    仅输出你的财务分析步骤和计算出的用于企业营运能力分析的财务指标。

    # 输入参数
    主要指标表：{input_sheet_1}
    现金流量表：{input_sheet_4}
    金融领域知识：{rag}

    ''',
    input_variables=['input']
)

# 根据template4的结果，进行偿债能力分析

template5 = PromptTemplate(
    template='''
    # 角色
    你是一位资深的财务分析师，善于通过企业的主要财务指标表、现金流量表来分析企业的营运能力。

    # 目标
    通过提供的主要财务指标表、资产负债表，结合你的金融领域知识，计算用于评估企业营运能力的财务指标，并与前两年的数据进行对比，从而揭示出企业的风险点所在。

    # 技能
    - 丰富的股票市场知识
    - 熟练的财务分析技能
    - 对财表隐藏风险点的洞察能力
    - 熟练的python数据分析能力

    # 工作流程
    1.接收财务指标计算结果和领域知识
    2.结合计算出的指标和你的知识，揭露企业在营运能力上可能出现的风险点。

    # 限制
    仅输出该企业的营运能力分析及该企业在营运能力上可能出现的风险点

    # 输入参数
    财务指标计算结果：{input_template2_result}
    金融领域对应计算指标：{rag}

    ''',
    input_variables=['input']
)

def model1(input):
    return model([HumanMessage(content=template1.format(input=input))]).content

def model2(input):
    return model([HumanMessage(content=template2.format(input=input))]).content

def model3(input):
    return model([HumanMessage(content=template3.format(input=input))]).content

def model4(input):
    return model([HumanMessage(content=template4.format(input=input))]).content

def model5(input):
    return model([HumanMessage(content=template5.format(input=input))]).content

# 封装成一个工作流
def calculate_operation_ability(input_data):

    # 第一步：模型1进行信息抽取
    calculation_result = model1(input_data)

    # 第二步：模型2进行信息抽取准确性判断规则制定
    rule_result = model2(calculation_result)

    # 第三步：模型3根据判断规则对抽取的信息进行研判和修正
    judge_result = model3(rule_result)

    # 第二步：模型4解释计算结果并进行营运能力分析
    explanation = model4(calculation_result)
    
    # 第三步：模型5解释计算结果并进行营运能力分析
    calculate = model5(explanation)

    # 返回最终结果
    return {
        "calculation_result": calculation_result,
        "rule_result": rule_result,
        "judge_result": judge_result,
        "explanation": explanation,
        "calculate": calculate
    }

# 使用工作流
input_data = '主要指标表数据+现金流量表数据'
results = calculate_operation_ability(input_data)
print("信息抽取结果:", results["judge_result"])
print("指标计算结果:", results["explanation"])
print("偿债能力分析:", results["calculate"])
