#!/usr/bin/env python3
"""
Generate multi-turn conversation samples with deliberate duplicates
for testing dedup pipeline on multi-turn dialogue scenarios.

Duplicate types:
  - Exact copies of multi-turn conversations
  - Near-duplicates: same topic/turns, slight wording changes
  - Partial overlaps: first N turns identical, later turns differ
  - Turn-count variants: same topic, but 2-turn vs 4-turn vs 6-turn
"""

import json
import random

random.seed(456)

OUTPUT = "./test_multiturn_duplicates.jsonl"


def conv(messages, system_text="你是一个有用的AI助手。"):
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": system_text}],
        "messages": [
            {"role": r, "content": [{"text": t}]}
            for r, t in messages
        ],
    }


samples = []

# ======================================================================
# Group A: "Python调试" — 3个完全相同的精确重复（多轮）
# ======================================================================
multiturn_debug = [
    ("user", "我的Python代码运行时报了一个TypeError,怎么排查？"),
    ("assistant",
     "TypeError通常是类型不匹配导致的。可以从以下几方面排查:\n"
     "1. 查看完整的错误堆栈,定位出错的行号\n"
     "2. 检查该行涉及的变量类型,用type()或isinstance()确认\n"
     "3. 常见原因:字符串和数字混合运算、None值参与运算、函数参数类型错误"),
    ("user", "堆栈信息显示是在一个字典操作的地方出错了,提示 'unhashable type: list'"),
    ("assistant",
     "这个错误是因为你用了列表(list)作为字典的key。字典的key必须是可哈希的类型,"
     "比如字符串、数字、元组等。列表是可变对象,不能作为key。\n\n"
     "解决方案:\n"
     "1. 如果列表内容固定,转成元组: tuple(your_list) 作为key\n"
     "2. 如果列表是某条记录的标识,考虑用字符串拼接: ','.join(map(str, your_list))\n"
     "3. 检查是否误把列表传给了需要hashable类型的地方"),
    ("user", "转成元组之后确实解决了,但是现在又报了KeyError"),
    ("assistant",
     "KeyError说明你访问的key在字典中不存在。几个排查方法:\n"
     "1. 用 dict.get(key, default) 代替 dict[key],避免报错\n"
     "2. 用 if key in dict 先检查key是否存在\n"
     "3. 打印所有key看看实际内容: print(list(dict.keys()))\n"
     "4. 注意key的类型是否一致——比如元组 (1,2) 和 (1, 2) 是相同的,"
     "但 ('1','2') 和 (1,2) 是不同的"),
]
# 精确重复 x3
samples.append(conv(multiturn_debug))
samples.append(conv(multiturn_debug))
samples.append(conv(multiturn_debug))

# ======================================================================
# Group B: "数据库设计" — 3个近似重复（措辞微调,多轮）
# ======================================================================
samples.append(conv([
    ("user", "如何设计一个好的数据库表结构？"),
    ("assistant",
     "设计数据库表结构需要遵循以下原则:\n"
     "1. 确定实体和关系: 先画ER图,理清表之间的一对一、一对多、多对多关系\n"
     "2. 遵循范式: 至少满足第三范式,减少数据冗余\n"
     "3. 合理选择主键: 推荐使用自增ID或UUID\n"
     "4. 字段类型要精确: 比如价格用DECIMAL而不是FLOAT"),
    ("user", "什么时候需要反范式设计？"),
    ("assistant",
     "反范式适用于以下场景:\n"
     "1. 读多写少的情况: 为了减少JOIN操作,可以冗余存储常用字段\n"
     "2. 报表查询: 预先计算并存储汇总数据,避免实时聚合\n"
     "3. 高并发场景: 减少表关联可以降低锁竞争\n"
     "但要注意: 冗余数据需要保持一致性,更新时要同步所有副本。"),
    ("user", "索引应该怎么建？"),
    ("assistant",
     "索引建立的原则:\n"
     "1. WHERE条件中频繁使用的列要建索引\n"
     "2. 联合索引遵循最左前缀原则\n"
     "3. 选择性高的列更适合建索引(如用户ID vs 性别)\n"
     "4. 避免过多索引,每个索引都增加写入开销\n"
     "5. 覆盖索引可以避免回表: 把查询需要的列都放进索引"),
]))

samples.append(conv([
    ("user", "如何设计一个好的数据库表结构？"),
    ("assistant",
     "设计数据库表结构需要遵循以下原则:\n"
     "1. 确定实体和关系: 先画ER图,理清表之间的一对一、一对多、多对多关系\n"
     "2. 遵循范式: 至少达到第三范式,减少数据冗余\n"
     "3. 合理选择主键: 推荐使用自增ID或UUID\n"
     "4. 字段类型要精确: 比如金额用DECIMAL而不是FLOAT"),
    ("user", "什么时候需要反范式设计？"),
    ("assistant",
     "反范式适用于以下场景:\n"
     "1. 读多写少的情况: 为了减少JOIN操作,可以冗余存储常用字段\n"
     "2. 报表查询: 提前计算并存储汇总数据,避免实时聚合\n"
     "3. 高并发场景: 减少表关联可以降低锁竞争\n"
     "但要注意: 冗余数据需要维持一致性,更新时要同步所有副本。"),
    ("user", "索引应该怎么建？"),
    ("assistant",
     "索引建立的原则:\n"
     "1. WHERE条件中频繁使用的列要建索引\n"
     "2. 联合索引遵循最左前缀原则\n"
     "3. 选择性高的列更适合建索引(如用户ID vs 性别)\n"
     "4. 避免过多索引,每个索引都会增加写入开销\n"
     "5. 覆盖索引可以避免回表: 把查询需要的列都包含在索引里"),
]))

samples.append(conv([
    ("user", "如何设计一个好的数据库表结构？"),
    ("assistant",
     "设计数据库表结构需要遵循以下原则:\n"
     "1. 确定实体和关系: 先画ER图,理清表之间的一对一、一对多、多对多关系\n"
     "2. 遵循范式: 至少满足第三范式,减少数据冗余\n"
     "3. 合理选择主键: 建议使用自增ID或UUID\n"
     "4. 字段类型要精确: 例如价格用DECIMAL而不是FLOAT"),
    ("user", "什么时候需要反范式设计？"),
    ("assistant",
     "反范式适用于以下场景:\n"
     "1. 读多写少的情况: 为了减少JOIN操作,可以冗余存储常用字段\n"
     "2. 报表查询: 预先计算并存储汇总数据,避免实时聚合\n"
     "3. 高并发场景: 减少表关联能够降低锁竞争\n"
     "但要注意: 冗余数据需要保持一致性,更新时必须同步所有副本。"),
    ("user", "索引应该怎么建？"),
    ("assistant",
     "索引建立的原则:\n"
     "1. WHERE条件中频繁使用的列要建索引\n"
     "2. 联合索引遵循最左前缀原则\n"
     "3. 选择性高的列更适合建索引(如用户ID vs 性别)\n"
     "4. 避免过多索引,每个索引都会带来写入开销\n"
     "5. 覆盖索引可以避免回表: 把查询涉及的列都放进索引"),
]))

# ======================================================================
# Group C: "前端面试" — 前两轮相同,第三轮不同（部分重叠多轮对话）
# ======================================================================
common_frontend_turns = [
    ("user", "前端面试中经常会问到闭包,能详细解释一下吗？"),
    ("assistant",
     "闭包是指一个函数能够访问其外部函数作用域中的变量,即使外部函数已经执行完毕。\n\n"
     "核心原理: JavaScript中函数创建时会保存对外部变量的引用,形成一个'封闭的环境'。\n\n"
     "经典例子:\n"
     "function counter() {\n"
     "  let count = 0;\n"
     "  return function() { return ++count; };\n"
     "}\n"
     "const inc = counter();\n"
     "inc(); // 1\n"
     "inc(); // 2\n\n"
     "这里内部函数引用了count变量,即使counter()已经返回,count仍然'活着'。"),
    ("user", "闭包有什么常见的应用场景？"),
    ("assistant",
     "闭包的常见应用:\n"
     "1. 数据封装/私有变量: 模拟面向对象中的private\n"
     "2. 函数工厂: 根据参数生成定制化函数\n"
     "3. 回调函数: 事件处理中保持对上下文的引用\n"
     "4. 柯里化: 将多参数函数转换为一系列单参数函数\n"
     "5. 模块模式: IIFE + 闭包实现模块化"),
]

# 变体C1: 追问关于内存泄漏
samples.append(conv(common_frontend_turns + [
    ("user", "闭包会不会导致内存泄漏？"),
    ("assistant",
     "闭包本身不会直接导致内存泄漏,但使用不当会造成内存问题:\n"
     "1. 如果闭包引用了大对象但不再使用,垃圾回收器无法释放它们\n"
     "2. 循环引用加上闭包可能导致旧版IE的内存泄漏\n"
     "3. 频繁创建闭包而不释放引用会占用大量内存\n\n"
     "预防方法: 不需要时手动置null、避免在循环中创建闭包、使用WeakMap/WeakRef。"),
]))

# 变体C2: 追问关于this指向
samples.append(conv(common_frontend_turns + [
    ("user", "闭包中的this指向有什么需要注意的？"),
    ("assistant",
     "闭包中的this是一个常见的坑:\n"
     "1. 普通函数中this取决于调用方式,不是定义时的作用域\n"
     "2. 箭头函数没有自己的this,会捕获外层的this——这在闭包中非常有用\n"
     "3. 经典问题: setTimeout中的回调函数this会指向window\n\n"
     "解决方案:\n"
     "- 使用箭头函数: () => { this.xxx }\n"
     "- 提前保存: const self = this;\n"
     "- 使用bind: function.bind(this)"),
]))

# 变体C3: 追问关于性能
samples.append(conv(common_frontend_turns + [
    ("user", "使用闭包对性能有影响吗？"),
    ("assistant",
     "闭包对性能的影响:\n"
     "1. 内存占用: 闭包会保持对外部变量的引用,占用额外内存\n"
     "2. 创建开销: 每次创建闭包都会分配新的作用域链,比普通函数稍慢\n"
     "3. 垃圾回收延迟: 被闭包引用的变量不会被立即回收\n\n"
     "但在现代JS引擎中,这些影响通常很小。V8等引擎对闭包做了大量优化,"
     "只有在极端情况(如百万级闭包)下才需要担心性能问题。"),
]))

# ======================================================================
# Group D: "微服务架构讨论" — 相同主题不同轮数（2轮 vs 4轮 vs 6轮）
# ======================================================================
samples.append(conv([
    ("user", "微服务架构和单体架构相比有什么优势？"),
    ("assistant",
     "微服务架构相比单体架构的优势:\n"
     "1. 独立部署: 每个服务可以单独发布,不影响其他服务\n"
     "2. 技术多样性: 不同服务可以使用不同的编程语言和技术栈\n"
     "3. 可扩展性: 可以针对热点服务单独扩容\n"
     "4. 故障隔离: 一个服务故障不会导致整个系统崩溃"),
]))

samples.append(conv([
    ("user", "微服务架构和单体架构相比有什么优势？"),
    ("assistant",
     "微服务架构相比单体架构的优势:\n"
     "1. 独立部署: 每个服务可以单独发布,不影响其他服务\n"
     "2. 技术多样性: 不同服务可以使用不同的编程语言和技术栈\n"
     "3. 可扩展性: 可以针对热点服务单独扩容\n"
     "4. 故障隔离: 一个服务故障不会导致整个系统崩溃"),
    ("user", "微服务有什么缺点吗？"),
    ("assistant",
     "微服务的主要挑战:\n"
     "1. 分布式复杂性: 网络延迟、服务发现、负载均衡都需要处理\n"
     "2. 数据一致性: 跨服务的事务管理比单体困难得多\n"
     "3. 运维成本: 需要容器化、服务编排、日志聚合、链路追踪等基础设施\n"
     "4. 接口管理: 服务间API的版本管理和兼容性要特别注意"),
]))

samples.append(conv([
    ("user", "微服务架构和单体架构相比有什么优势？"),
    ("assistant",
     "微服务架构相比单体架构的优势:\n"
     "1. 独立部署: 每个服务可以单独发布,不影响其他服务\n"
     "2. 技术多样性: 不同服务可以使用不同的编程语言和技术栈\n"
     "3. 可扩展性: 可以针对热点服务单独扩容\n"
     "4. 故障隔离: 一个服务故障不会导致整个系统崩溃"),
    ("user", "微服务有什么缺点吗？"),
    ("assistant",
     "微服务的主要挑战:\n"
     "1. 分布式复杂性: 网络延迟、服务发现、负载均衡都需要处理\n"
     "2. 数据一致性: 跨服务的事务管理比单体困难得多\n"
     "3. 运维成本: 需要容器化、服务编排、日志聚合、链路追踪等基础设施\n"
     "4. 接口管理: 服务间API的版本管理和兼容性要特别注意"),
    ("user", "那什么时候应该选择微服务？"),
    ("assistant",
     "选择微服务的时机:\n"
     "1. 团队规模较大(>10人): 多团队并行开发时,微服务能减少代码冲突\n"
     "2. 业务复杂度高: 不同业务模块有明显的边界\n"
     "3. 需要弹性扩缩: 某些模块流量大但其他模块不需要扩容\n"
     "4. 技术演进需求: 想逐步替换遗留系统的部分模块\n\n"
     "如果是小团队或初创项目,建议从单体开始,等业务稳定后再拆分。"),
]))

# ======================================================================
# Group E: "Git工作流" — 近似重复多轮对话（措辞微调）
# ======================================================================
samples.append(conv([
    ("user", "Git分支管理有什么好的实践？"),
    ("assistant",
     "Git分支管理的最佳实践:\n"
     "1. 主分支保护: main/master分支设为受保护,只能通过PR合并\n"
     "2. 功能分支: 每个功能或修复使用独立的feature/fix分支\n"
     "3. 命名规范: feature/xxx, bugfix/xxx, hotfix/xxx\n"
     "4. 及时清理: 合并后删除已完成的分支"),
    ("user", "Git Flow和GitHub Flow有什么区别？"),
    ("assistant",
     "Git Flow:\n"
     "- 有develop、release、hotfix等多种分支类型\n"
     "- 适合有明确版本发布周期的项目\n"
     "- 流程较复杂,分支较多\n\n"
     "GitHub Flow:\n"
     "- 只有main和feature分支\n"
     "- 通过PR进行代码审查和合并\n"
     "- 适合持续部署的项目\n"
     "- 流程简单,更适合小团队"),
    ("user", "如何处理合并冲突？"),
    ("assistant",
     "处理合并冲突的步骤:\n"
     "1. git merge或git rebase时出现冲突,Git会标记冲突文件\n"
     "2. 打开冲突文件,找到<<<<<<< ======= >>>>>>>标记\n"
     "3. 手动编辑,保留正确的代码\n"
     "4. git add标记已解决的文件\n"
     "5. git commit完成合并\n\n"
     "建议: 经常从主分支拉取更新,减少冲突的可能性。"),
]))

samples.append(conv([
    ("user", "Git分支管理有什么好的实践？"),
    ("assistant",
     "Git分支管理的最佳实践:\n"
     "1. 主分支保护: main/master分支设为受保护,只能通过PR合并\n"
     "2. 功能分支: 每个功能或修复使用独立的feature/fix分支\n"
     "3. 命名规范: feature/xxx, bugfix/xxx, hotfix/xxx\n"
     "4. 及时清理: 合并后删掉已完成的分支"),
    ("user", "Git Flow和GitHub Flow有什么区别？"),
    ("assistant",
     "Git Flow:\n"
     "- 有develop、release、hotfix等多种分支类型\n"
     "- 适合有明确版本发布节奏的项目\n"
     "- 流程较复杂,分支较多\n\n"
     "GitHub Flow:\n"
     "- 只有main和feature分支\n"
     "- 通过PR进行代码评审和合并\n"
     "- 适合持续部署的项目\n"
     "- 流程简洁,更适合小团队"),
    ("user", "如何处理合并冲突？"),
    ("assistant",
     "处理合并冲突的步骤:\n"
     "1. git merge或git rebase时出现冲突,Git会标记冲突文件\n"
     "2. 打开冲突文件,找到<<<<<<< ======= >>>>>>>标记\n"
     "3. 手动编辑,保留正确的内容\n"
     "4. git add标记已解决的文件\n"
     "5. git commit完成合并\n\n"
     "建议: 经常从主分支同步更新,减少冲突的几率。"),
]))

# ======================================================================
# Group F: "算法复杂度" — 精确重复 x2 + 措辞微调 x1（多轮）
# ======================================================================
complexity_turns = [
    ("user", "什么是时间复杂度和空间复杂度？"),
    ("assistant",
     "时间复杂度衡量算法执行所需的时间随输入规模增长的变化趋势,用大O表示法。\n"
     "空间复杂度衡量算法执行所需的额外内存空间。\n\n"
     "常见时间复杂度(从快到慢):\n"
     "O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n)"),
    ("user", "能举一些具体算法的例子吗？"),
    ("assistant",
     "各种复杂度的典型算法:\n"
     "- O(1): 数组按索引访问、哈希表查找\n"
     "- O(log n): 二分查找、平衡二叉树操作\n"
     "- O(n): 线性搜索、遍历数组\n"
     "- O(n log n): 快速排序、归并排序\n"
     "- O(n²): 冒泡排序、选择排序\n"
     "- O(2^n): 暴力递归求子集"),
    ("user", "实际开发中怎么分析代码的复杂度？"),
    ("assistant",
     "分析复杂度的方法:\n"
     "1. 看循环嵌套: 单层循环O(n),双层嵌套O(n²)\n"
     "2. 递归用递推式: T(n) = 2T(n/2) + O(n) → O(n log n)\n"
     "3. 取最高项: O(n² + n) = O(n²)\n"
     "4. 注意隐藏的复杂度: 字符串拼接在某些语言中是O(n)\n"
     "5. 实际项目中可以用profiler工具测量真实性能"),
]
# 精确重复 x2
samples.append(conv(complexity_turns))
samples.append(conv(complexity_turns))
# 措辞微调 x1
samples.append(conv([
    ("user", "什么是时间复杂度和空间复杂度？"),
    ("assistant",
     "时间复杂度衡量算法运行所需的时间随输入规模增长的变化趋势,用大O表示法。\n"
     "空间复杂度衡量算法运行所需的额外内存空间。\n\n"
     "常见时间复杂度(从快到慢):\n"
     "O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n)"),
    ("user", "能举一些具体算法的例子吗？"),
    ("assistant",
     "各种复杂度的典型算法:\n"
     "- O(1): 数组按索引访问、哈希表查找\n"
     "- O(log n): 二分查找、平衡二叉树操作\n"
     "- O(n): 线性搜索、遍历数组\n"
     "- O(n log n): 快速排序、归并排序\n"
     "- O(n²): 冒泡排序、选择排序\n"
     "- O(2^n): 暴力递归求子集"),
    ("user", "实际开发中怎么分析代码的复杂度？"),
    ("assistant",
     "分析复杂度的方法:\n"
     "1. 看循环嵌套: 单层循环O(n),双层嵌套O(n²)\n"
     "2. 递归用递推式: T(n) = 2T(n/2) + O(n) → O(n log n)\n"
     "3. 取最高项: O(n² + n) = O(n²)\n"
     "4. 注意隐藏的复杂度: 字符串拼接在某些语言中是O(n)\n"
     "5. 实际项目中可以用性能分析工具测量真实性能"),
]))

# ======================================================================
# Group G: "Docker部署" — 4轮对话 x3，前3轮近似、第4轮完全不同
# ======================================================================
samples.append(conv([
    ("user", "如何用Docker部署一个Python Web应用？"),
    ("assistant",
     "基本步骤:\n"
     "1. 编写Dockerfile\n"
     "2. 构建镜像: docker build -t myapp .\n"
     "3. 运行容器: docker run -p 8080:8080 myapp\n\n"
     "一个Flask应用的Dockerfile示例:\n"
     "FROM python:3.11-slim\n"
     "WORKDIR /app\n"
     "COPY requirements.txt .\n"
     "RUN pip install -r requirements.txt\n"
     "COPY . .\n"
     "CMD [\"python\", \"app.py\"]"),
    ("user", "如何优化Docker镜像的大小？"),
    ("assistant",
     "优化Docker镜像大小的方法:\n"
     "1. 使用alpine或slim基础镜像\n"
     "2. 多阶段构建: 编译阶段和运行阶段分开\n"
     "3. 合并RUN指令减少层数\n"
     "4. 使用.dockerignore排除不需要的文件\n"
     "5. 清理缓存: pip install --no-cache-dir"),
    ("user", "如何用docker-compose管理多个容器？"),
    ("assistant",
     "docker-compose.yml示例:\n"
     "version: '3.8'\n"
     "services:\n"
     "  web:\n"
     "    build: .\n"
     "    ports: ['8080:8080']\n"
     "    depends_on: [db]\n"
     "  db:\n"
     "    image: postgres:15\n"
     "    environment:\n"
     "      POSTGRES_PASSWORD: secret\n\n"
     "常用命令:\n"
     "docker-compose up -d  # 后台启动\n"
     "docker-compose logs   # 查看日志\n"
     "docker-compose down   # 停止并删除"),
    ("user", "生产环境中Docker有什么注意事项？"),
    ("assistant",
     "生产环境Docker注意事项:\n"
     "1. 不要用root用户运行容器,创建专用用户\n"
     "2. 固定基础镜像版本,不要用latest标签\n"
     "3. 设置健康检查(HEALTHCHECK)\n"
     "4. 限制容器资源(--memory, --cpus)\n"
     "5. 使用Docker secrets管理敏感信息\n"
     "6. 定期更新基础镜像修补安全漏洞"),
]))

samples.append(conv([
    ("user", "如何用Docker部署一个Python Web应用？"),
    ("assistant",
     "基本步骤:\n"
     "1. 编写Dockerfile\n"
     "2. 构建镜像: docker build -t myapp .\n"
     "3. 运行容器: docker run -p 8080:8080 myapp\n\n"
     "一个Flask应用的Dockerfile示例:\n"
     "FROM python:3.11-slim\n"
     "WORKDIR /app\n"
     "COPY requirements.txt .\n"
     "RUN pip install -r requirements.txt\n"
     "COPY . .\n"
     "CMD [\"python\", \"app.py\"]"),
    ("user", "如何优化Docker镜像的大小？"),
    ("assistant",
     "优化Docker镜像大小的方法:\n"
     "1. 使用alpine或slim基础镜像\n"
     "2. 多阶段构建: 编译阶段和运行阶段分开\n"
     "3. 合并RUN指令减少层数\n"
     "4. 使用.dockerignore排除不需要的文件\n"
     "5. 清理缓存: pip install --no-cache-dir"),
    ("user", "如何用docker-compose管理多个容器？"),
    ("assistant",
     "docker-compose.yml示例:\n"
     "version: '3.8'\n"
     "services:\n"
     "  web:\n"
     "    build: .\n"
     "    ports: ['8080:8080']\n"
     "    depends_on: [db]\n"
     "  db:\n"
     "    image: postgres:15\n"
     "    environment:\n"
     "      POSTGRES_PASSWORD: secret\n\n"
     "常用命令:\n"
     "docker-compose up -d  # 后台启动\n"
     "docker-compose logs   # 查看日志\n"
     "docker-compose down   # 停止并删除"),
    ("user", "容器内如何调试问题？"),
    ("assistant",
     "容器调试方法:\n"
     "1. docker exec -it container_id /bin/bash 进入容器\n"
     "2. docker logs container_id 查看日志\n"
     "3. docker inspect 查看容器详细信息\n"
     "4. docker stats 监控资源使用\n"
     "5. 挂载volume映射源码,方便本地修改实时生效"),
]))

samples.append(conv([
    ("user", "如何用Docker部署一个Python Web应用？"),
    ("assistant",
     "基本步骤:\n"
     "1. 编写Dockerfile\n"
     "2. 构建镜像: docker build -t myapp .\n"
     "3. 运行容器: docker run -p 8080:8080 myapp\n\n"
     "一个Flask应用的Dockerfile示例:\n"
     "FROM python:3.11-slim\n"
     "WORKDIR /app\n"
     "COPY requirements.txt .\n"
     "RUN pip install -r requirements.txt\n"
     "COPY . .\n"
     "CMD [\"python\", \"app.py\"]"),
    ("user", "如何优化Docker镜像的大小？"),
    ("assistant",
     "优化Docker镜像大小的技巧:\n"
     "1. 使用alpine或slim基础镜像\n"
     "2. 多阶段构建: 编译阶段和运行阶段分离\n"
     "3. 合并RUN指令减少层数\n"
     "4. 使用.dockerignore排除不需要的文件\n"
     "5. 清理安装缓存: pip install --no-cache-dir"),
    ("user", "如何用docker-compose管理多个容器？"),
    ("assistant",
     "docker-compose.yml示例:\n"
     "version: '3.8'\n"
     "services:\n"
     "  web:\n"
     "    build: .\n"
     "    ports: ['8080:8080']\n"
     "    depends_on: [db]\n"
     "  db:\n"
     "    image: postgres:15\n"
     "    environment:\n"
     "      POSTGRES_PASSWORD: secret\n\n"
     "常用命令:\n"
     "docker-compose up -d  # 后台启动\n"
     "docker-compose logs   # 查看日志\n"
     "docker-compose down   # 停止并删除"),
    ("user", "如何实现容器的自动重启？"),
    ("assistant",
     "容器自动重启策略:\n"
     "1. docker run --restart=always  # 总是重启\n"
     "2. docker run --restart=unless-stopped  # 除非手动停止\n"
     "3. docker run --restart=on-failure:5  # 失败时重启,最多5次\n\n"
     "在docker-compose中:\n"
     "services:\n"
     "  web:\n"
     "    restart: unless-stopped\n\n"
     "生产环境建议用Kubernetes管理,它提供更强大的自愈能力。"),
]))

# ======================================================================
# Group H: "Python异步编程" — 精确重复2轮对话 + 一个近似版本
# ======================================================================
async_turns = [
    ("user", "Python的async/await是怎么工作的？"),
    ("assistant",
     "Python的异步编程基于事件循环(event loop):\n\n"
     "1. async def 定义协程函数\n"
     "2. await 挂起当前协程,等待异步操作完成\n"
     "3. 事件循环调度多个协程并发执行\n\n"
     "核心概念:\n"
     "- 协程(coroutine): async def定义的函数\n"
     "- 任务(task): asyncio.create_task()创建的可调度单元\n"
     "- Future: 代表异步操作的最终结果"),
    ("user", "和多线程相比有什么区别？"),
    ("assistant",
     "异步 vs 多线程:\n\n"
     "异步(asyncio):\n"
     "- 单线程,通过协程切换实现并发\n"
     "- 没有线程安全问题,不需要锁\n"
     "- 适合I/O密集型: 网络请求、文件读写\n"
     "- 不能利用多核CPU\n\n"
     "多线程(threading):\n"
     "- 多线程,受GIL限制同时只能一个线程执行Python字节码\n"
     "- 需要处理线程安全问题\n"
     "- 也适合I/O密集型\n"
     "- 切换开销比协程大\n\n"
     "CPU密集型任务建议用multiprocessing(多进程)。"),
]
samples.append(conv(async_turns))
samples.append(conv(async_turns))
samples.append(conv([
    ("user", "Python的async/await是怎么工作的？"),
    ("assistant",
     "Python的异步编程基于事件循环(event loop):\n\n"
     "1. async def 定义协程函数\n"
     "2. await 挂起当前协程,等待异步操作完成\n"
     "3. 事件循环负责调度多个协程并发运行\n\n"
     "核心概念:\n"
     "- 协程(coroutine): async def定义的函数\n"
     "- 任务(task): asyncio.create_task()创建的可调度单元\n"
     "- Future: 代表异步操作的最终结果"),
    ("user", "和多线程相比有什么区别？"),
    ("assistant",
     "异步 vs 多线程:\n\n"
     "异步(asyncio):\n"
     "- 单线程,通过协程切换实现并发\n"
     "- 没有线程安全问题,不需要加锁\n"
     "- 适合I/O密集型: 网络请求、文件读写\n"
     "- 不能利用多核CPU\n\n"
     "多线程(threading):\n"
     "- 多线程,受GIL限制同时只能一个线程执行Python字节码\n"
     "- 需要处理线程安全问题\n"
     "- 也适合I/O密集型\n"
     "- 上下文切换开销比协程大\n\n"
     "CPU密集型任务建议用multiprocessing(多进程)。"),
]))

# ======================================================================
# Unique multi-turn samples (10 entries, no duplicates)
# ======================================================================
unique_conversations = [
    [("user", "能介绍一下Redis的持久化机制吗？"),
     ("assistant",
      "Redis有两种持久化方式:\n"
      "1. RDB(快照): 定时将内存数据保存到磁盘,恢复快但可能丢失最后一次快照后的数据\n"
      "2. AOF(追加日志): 记录每次写操作,数据更安全但文件更大"),
     ("user", "生产环境应该用哪种？"),
     ("assistant",
      "生产环境建议RDB和AOF同时开启:\n"
      "- AOF保证数据安全性(设置appendfsync everysec)\n"
      "- RDB用于灾难恢复和快速重启\n"
      "- Redis 4.0+还支持混合持久化: RDB快照+增量AOF")],

    [("user", "什么是CORS？前端遇到跨域问题怎么解决？"),
     ("assistant",
      "CORS是浏览器的同源策略限制,当请求的协议、域名或端口与当前页面不同时会被拦截。"),
     ("user", "后端怎么配置？"),
     ("assistant",
      "以Express为例:\n"
      "const cors = require('cors');\n"
      "app.use(cors({ origin: 'http://localhost:3000' }));\n\n"
      "也可以手动设置响应头:\n"
      "res.setHeader('Access-Control-Allow-Origin', '*');\n"
      "res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE');")],

    [("user", "如何用Python实现一个简单的爬虫？"),
     ("assistant",
      "用requests + BeautifulSoup即可:\n"
      "import requests\n"
      "from bs4 import BeautifulSoup\n"
      "resp = requests.get(url)\n"
      "soup = BeautifulSoup(resp.text, 'html.parser')\n"
      "titles = soup.find_all('h2')"),
     ("user", "如果网站需要登录怎么办？"),
     ("assistant",
      "处理登录的方法:\n"
      "1. 使用Session保持cookie: s = requests.Session(); s.post(login_url, data=...)\n"
      "2. 直接在header中添加Cookie或Token\n"
      "3. 对于JS渲染的页面,用Selenium或Playwright模拟浏览器")],

    [("user", "Python中如何处理大文件？"),
     ("assistant",
      "处理大文件的关键是避免一次性读入内存:\n"
      "1. 逐行读取: for line in open('file.txt')\n"
      "2. 分块读取: file.read(chunk_size)\n"
      "3. 用pandas分块: pd.read_csv('big.csv', chunksize=10000)\n"
      "4. 用mmap内存映射: mmap.mmap(f.fileno(), 0)"),
     ("user", "如果要处理几百GB的日志文件呢？"),
     ("assistant",
      "超大文件处理策略:\n"
      "1. 用生成器配合流式处理,内存占用恒定\n"
      "2. 多进程并行处理: 先split -l分割文件,再并行处理\n"
      "3. 使用专门工具: awk/sed命令行预处理\n"
      "4. 考虑Spark/Dask等分布式计算框架")],

    [("user", "什么是JWT？怎么用它做认证？"),
     ("assistant",
      "JWT(JSON Web Token)由三部分组成: Header.Payload.Signature\n"
      "流程: 用户登录→服务端生成JWT→客户端存储→每次请求携带JWT→服务端验证"),
     ("user", "JWT有什么安全问题？"),
     ("assistant",
      "JWT的安全注意事项:\n"
      "1. 不要在Payload中放敏感信息,它是Base64编码不是加密\n"
      "2. 设置合理的过期时间\n"
      "3. 使用HTTPS传输\n"
      "4. 签名算法选择RS256或HS256\n"
      "5. 考虑Token刷新机制和黑名单方案")],

    [("user", "什么是WebSocket？和HTTP有什么区别？"),
     ("assistant",
      "WebSocket是全双工通信协议,建立连接后服务端可以主动推送消息。\n"
      "HTTP是请求-响应模式,客户端发请求服务端才能回复。"),
     ("user", "什么场景下需要用WebSocket？"),
     ("assistant",
      "适合WebSocket的场景:\n"
      "1. 即时通讯: 聊天应用\n"
      "2. 实时推送: 股票行情、体育比分\n"
      "3. 协同编辑: 多人同时编辑文档\n"
      "4. 游戏: 需要低延迟的实时交互")],

    [("user", "MySQL和PostgreSQL该怎么选？"),
     ("assistant",
      "MySQL: 简单易用,生态成熟,适合Web应用。\n"
      "PostgreSQL: 功能更丰富,支持JSON、数组、自定义类型,适合复杂查询。"),
     ("user", "性能方面呢？"),
     ("assistant",
      "性能差异:\n"
      "- 简单查询: MySQL通常更快\n"
      "- 复杂查询/分析: PostgreSQL优化器更强\n"
      "- 并发写入: PostgreSQL的MVCC实现更优\n"
      "- 关键是根据业务场景选择,两者性能差距在大多数场景下不是瓶颈")],

    [("user", "如何设计一个短链接系统？"),
     ("assistant",
      "短链接系统设计:\n"
      "1. 生成短码: Base62编码(a-z,A-Z,0-9),6位可表示56亿组合\n"
      "2. 存储映射: 短码→原始URL存Redis+数据库\n"
      "3. 跳转: 302重定向到原始URL"),
     ("user", "如何保证短码不重复？"),
     ("assistant",
      "防重复方案:\n"
      "1. 自增ID+Base62: 数据库自增ID转Base62,天然不重复\n"
      "2. 哈希+冲突检测: MD5/SHA取前6位,冲突则加盐重试\n"
      "3. 预生成: 提前批量生成短码存入池中,按需取用\n"
      "推荐方案1,简单可靠。")],

    [("user", "Python有哪些常用的设计模式？"),
     ("assistant",
      "Python常用设计模式:\n"
      "1. 单例: 用__new__或模块级变量\n"
      "2. 工厂: 函数返回不同类的实例\n"
      "3. 装饰器: Python原生支持\n"
      "4. 观察者: 用回调函数列表\n"
      "5. 策略: 传入不同的函数/类"),
     ("user", "Python中单例模式怎么实现最优雅？"),
     ("assistant",
      "最Python风格的单例实现:\n\n"
      "# 方式1: 模块本身就是单例\n"
      "# config.py\n"
      "settings = {'key': 'value'}\n\n"
      "# 方式2: 用装饰器\n"
      "def singleton(cls):\n"
      "    instances = {}\n"
      "    def get_instance(*args, **kwargs):\n"
      "        if cls not in instances:\n"
      "            instances[cls] = cls(*args, **kwargs)\n"
      "        return instances[cls]\n"
      "    return get_instance")],

    [("user", "什么是CI/CD？怎么搭建？"),
     ("assistant",
      "CI(持续集成): 代码提交后自动构建和测试。\n"
      "CD(持续部署): 测试通过后自动部署到生产环境。"),
     ("user", "用GitHub Actions怎么实现？"),
     ("assistant",
      "GitHub Actions示例(.github/workflows/ci.yml):\n"
      "name: CI\n"
      "on: [push, pull_request]\n"
      "jobs:\n"
      "  test:\n"
      "    runs-on: ubuntu-latest\n"
      "    steps:\n"
      "      - uses: actions/checkout@v4\n"
      "      - uses: actions/setup-python@v5\n"
      "      - run: pip install -r requirements.txt\n"
      "      - run: pytest")],
]
for msgs in unique_conversations:
    samples.append(conv(msgs))

# ── Shuffle ──
random.shuffle(samples)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ── Stats ──
print(f"Generated {len(samples)} multi-turn samples -> {OUTPUT}")
print()
print("Duplicate groups:")
print("  Group A: Python调试 (精确重复x3)          — 6 turns, exact")
print("  Group B: 数据库设计 (近似重复x3)          — 6 turns, overlap")
print("  Group C: 前端闭包   (前4 turns相同,后续不同x3) — 6 turns, partial")
print("  Group D: 微服务架构 (2/4/6 turns)         — 不同轮数")
print("  Group E: Git工作流  (近似重复x2)          — 6 turns, overlap")
print("  Group F: 算法复杂度 (精确x2 + 近似x1)    — 6 turns, mixed")
print("  Group G: Docker部署 (前6 turns近似,后2不同x3) — 8 turns, partial")
print("  Group H: Python异步 (精确x2 + 近似x1)    — 4 turns, mixed")
print(f"  Unique:  {len(unique_conversations)} entries")
print(f"  Total:   {len(samples)}")
