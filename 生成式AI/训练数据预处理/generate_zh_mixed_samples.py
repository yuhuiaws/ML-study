#!/usr/bin/env python3
"""Generate 50 Chinese and 50 Chinese-English mixed Bedrock-format conversation samples."""

import json

SCHEMA = "bedrock-conversation-2024"
SYSTEM_ZH = [{"text": "你是一个有用的对话助手。请进行自然、友好的对话。"}]
SYSTEM_MIX = [{"text": "You are a helpful bilingual assistant. Engage in natural dialogue, mixing Chinese and English as appropriate."}]


def msg(role, text):
    return {"role": role, "content": [{"text": text}]}


def conv(system, messages):
    return {"schemaVersion": SCHEMA, "system": system, "messages": messages}


# ============================================================
# 50 Chinese conversations
# ============================================================
zh_conversations = [
    # 1
    [msg("user", "你好，请问附近有什么好吃的餐厅吗？"),
     msg("assistant", "你好！附近有一家川菜馆非常不错，他们的麻婆豆腐特别地道。走路大概五分钟就到了。")],
    # 2
    [msg("user", "今天天气怎么样？"),
     msg("assistant", "今天天气很好，阳光明媚，气温大概25度左右，非常适合出去走走。")],
    # 3
    [msg("user", "我想学做饭，你有什么建议吗？"),
     msg("assistant", "建议你先从简单的家常菜开始，比如番茄炒蛋、蒜蓉西兰花这些。可以看一些烹饪教学视频，跟着一步步做。"),
     msg("user", "番茄炒蛋看起来确实挺简单的。"),
     msg("assistant", "是的，关键是火候和调味。先把鸡蛋打散炒熟盛出来，再炒番茄，最后把蛋倒回去翻炒均匀就行了。")],
    # 4
    [msg("user", "最近工作压力好大，你有什么减压的方法吗？"),
     msg("assistant", "我能理解工作压力大的感觉。你可以试试运动，比如跑步或者瑜伽，也可以听听音乐放松一下。"),
     msg("user", "我平时没什么时间运动。"),
     msg("assistant", "那你可以试试冥想，每天十分钟就够了。睡前做几次深呼吸也能帮助缓解压力。")],
    # 5
    [msg("user", "请问这趟高铁到上海要多长时间？"),
     msg("assistant", "这趟高铁到上海大约需要四个半小时，中途会停靠南京和苏州。"),
     msg("user", "好的，那我需要提前多久到车站？"),
     msg("assistant", "建议至少提前半小时到，这样有足够的时间过安检和找站台。")],
    # 6
    [msg("user", "你觉得养猫好还是养狗好？"),
     msg("assistant", "这取决于你的生活方式。如果你工作比较忙，猫可能更适合你，因为猫比较独立。如果你喜欢户外活动，狗会是很好的伙伴。")],
    # 7
    [msg("user", "我的手机屏幕碎了，去哪里修比较好？"),
     msg("assistant", "你可以去官方售后服务中心修，质量比较有保障。如果想省钱的话，也可以去手机维修店，但要注意选择口碑好的店铺。"),
     msg("user", "官方售后大概要多少钱？"),
     msg("assistant", "具体价格要看手机型号，一般来说更换原装屏幕的费用在几百到一千多不等。你可以先在官网查一下报价。")],
    # 8
    [msg("user", "周末有什么好玩的地方推荐吗？"),
     msg("assistant", "如果你喜欢自然风景，可以去郊外的公园徒步。如果喜欢文化类的，市中心新开了一个艺术展览，评价很好。"),
     msg("user", "艺术展览听起来不错，需要提前买票吗？"),
     msg("assistant", "最好提前在网上买票，周末去的人比较多，现场排队可能要等很久。")],
    # 9
    [msg("user", "我家孩子马上要上小学了，你觉得需要提前学些什么？"),
     msg("assistant", "其实不需要提前学太多课本知识，更重要的是培养好的学习习惯和自理能力。比如能自己整理书包、按时完成任务这些。"),
     msg("user", "那拼音和算术呢？"),
     msg("assistant", "适当接触一些是可以的，但不要给孩子太大压力。保持对学习的兴趣才是最重要的。")],
    # 10
    [msg("user", "你好，我想开一个银行账户。"),
     msg("assistant", "好的，请问您要开个人储蓄账户还是信用卡账户？"),
     msg("user", "个人储蓄账户。"),
     msg("assistant", "好的，请您带好身份证和一张一寸照片，到柜台办理就可以了。整个过程大概十五分钟。")],
    # 11
    [msg("user", "最近有什么好看的电影推荐吗？"),
     msg("assistant", "最近上映的那部科幻片评分很高，讲的是未来世界的故事，视觉效果特别震撼。你喜欢什么类型的电影？"),
     msg("user", "我比较喜欢悬疑片。"),
     msg("assistant", "那你可以看看上个月上映的那部悬疑推理片，剧情反转很精彩，很多观众都说没猜到结局。")],
    # 12
    [msg("user", "请问这件衣服有别的颜色吗？"),
     msg("assistant", "有的，这款除了您看到的黑色，还有白色、藏蓝色和卡其色。"),
     msg("user", "藏蓝色的能让我试一下吗？"),
     msg("assistant", "当然可以，请稍等，我给您拿一件。试衣间在那边。")],
    # 13
    [msg("user", "我刚搬到这个城市，怎么办公交卡？"),
     msg("assistant", "你可以去地铁站的服务中心办理，带上身份证就行。也可以直接用手机上的交通卡功能，绑定之后刷手机就能乘车。")],
    # 14
    [msg("user", "医生，我最近总是头疼，怎么回事？"),
     msg("assistant", "头疼的原因有很多，可能跟睡眠不足、压力大或者用眼过度有关。你最近作息规律吗？"),
     msg("user", "确实最近经常熬夜。"),
     msg("assistant", "那很可能是睡眠不足引起的。建议你调整作息，尽量在晚上十一点前睡觉，如果症状持续的话再来做个详细检查。")],
    # 15
    [msg("user", "你好，我想预约下周三下午的会议室。"),
     msg("assistant", "好的，请问您需要几点到几点的？大会议室还是小会议室？"),
     msg("user", "下午两点到四点，大概十个人参加。"),
     msg("assistant", "那给您安排中型会议室就可以了，下周三下午两点到四点，我已经帮您预约好了。")],
    # 16
    [msg("user", "这道菜怎么做的？味道太好了！"),
     msg("assistant", "谢谢夸奖！这道红烧排骨先用冰糖炒出糖色，然后加入料酒、生抽和老抽，小火慢炖一个小时就好了。"),
     msg("user", "听起来不难，我回去也试试。"),
     msg("assistant", "关键是要有耐心，小火慢炖才能入味。肉要选带一点肥的，口感会更好。")],
    # 17
    [msg("user", "我的快递显示已签收，但是我没收到啊。"),
     msg("assistant", "别着急，有时候快递员会把包裹放在快递柜或者交给物业。你先去看看快递柜和物业那边有没有。"),
     msg("user", "我去看过了，都没有。"),
     msg("assistant", "那你可以打快递公司的客服电话投诉，让他们查一下派送员的记录。如果确实丢件了，可以要求赔偿。")],
    # 18
    [msg("user", "你觉得现在买房合适吗？"),
     msg("assistant", "买房是一个很大的决定，要考虑很多因素。你的经济状况、工作稳定性、所在城市的房价走势都很重要。如果是刚需自住，价格合适就可以考虑。")],
    # 19
    [msg("user", "我想给妈妈过生日，你有什么好的建议吗？"),
     msg("assistant", "可以为她准备一顿家庭聚餐，亲手做几道她喜欢吃的菜。如果时间允许，还可以带她去旅游或者泡温泉放松一下。"),
     msg("user", "她平时不太愿意出远门。"),
     msg("assistant", "那就在家里布置一下，买个蛋糕，一家人一起吃顿饭。陪伴对妈妈来说就是最好的礼物。")],
    # 20
    [msg("user", "请问这个小区的物业费是多少？"),
     msg("assistant", "我们小区的物业费是每月三块五一平方米，包含公共区域的清洁、绿化和安保服务。"),
     msg("user", "停车费另算吗？"),
     msg("assistant", "是的，地上停车位每月两百，地下停车位每月三百五。业主可以优先购买固定车位。")],
    # 21
    [msg("user", "你平时都怎么锻炼身体？"),
     msg("assistant", "我每周会跑步三到四次，每次大概五公里。周末有空的话还会去爬山或者骑自行车。"),
     msg("user", "我也想开始跑步，但总是坚持不下来。"),
     msg("assistant", "刚开始不要给自己定太高的目标，可以先从每次跑两公里开始，慢慢增加距离。找个朋友一起跑也更容易坚持。")],
    # 22
    [msg("user", "老师，我这次考试没考好，怎么办？"),
     msg("assistant", "别太沮丧，一次考试说明不了什么。重要的是分析一下哪些地方做错了，是知识点没掌握还是粗心大意。"),
     msg("user", "主要是最后两道大题没做出来。"),
     msg("assistant", "那说明你在这类题型上还需要多练习。建议你把类似的题目集中起来做，找到解题思路，慢慢就会好的。")],
    # 23
    [msg("user", "你好，我想退这件商品。"),
     msg("assistant", "好的，请问是什么原因要退货呢？"),
     msg("user", "尺码不合适，穿着太大了。"),
     msg("assistant", "好的，只要在七天内且不影响二次销售，都可以退。请您出示一下购物小票，我帮您办理退货手续。")],
    # 24
    [msg("user", "最近失眠严重，有什么好方法吗？"),
     msg("assistant", "失眠的话，首先要注意睡眠环境，保持卧室安静、黑暗。睡前不要看手机，可以泡个脚或者喝杯温牛奶。"),
     msg("user", "我试过喝牛奶，好像没什么效果。"),
     msg("assistant", "每个人情况不一样。你可以试试睡前听一些舒缓的音乐，或者做一些简单的拉伸运动。如果长期失眠的话，建议去看看医生。")],
    # 25
    [msg("user", "请问去机场怎么走最方便？"),
     msg("assistant", "从这里去机场最方便的是坐地铁，二号线转机场快线，全程大概四十分钟。"),
     msg("user", "打车的话要多久？"),
     msg("assistant", "不堵车的话大概半小时，但如果是早高峰的话可能要一个多小时。建议你留够时间。")],
    # 26
    [msg("user", "你觉得学一门新语言难吗？"),
     msg("assistant", "刚开始的时候会觉得难，但只要坚持下去就会越来越顺。关键是要每天花一点时间练习，多听多说多读。"),
     msg("user", "我想学日语，从哪里开始比较好？"),
     msg("assistant", "先从五十音图开始，把假名记熟。然后可以用教材配合语言学习的应用软件，循序渐进地学习。看日剧和动漫也是很好的辅助方式。")],
    # 27
    [msg("user", "这台电脑配置怎么样？适合做视频剪辑吗？"),
     msg("assistant", "这台电脑的处理器和显卡都不错，内存16G，基本的视频剪辑是够用的。如果你经常处理4K视频的话，建议升级到32G内存。")],
    # 28
    [msg("user", "师傅，我的车空调不制冷了，是怎么回事？"),
     msg("assistant", "空调不制冷一般有几种原因，可能是制冷剂不足、压缩机故障或者冷凝器堵塞。我先给您检查一下。"),
     msg("user", "大概要修多久？"),
     msg("assistant", "如果只是加氟的话，半个小时就搞定了。如果是其他问题，可能需要一到两天。我检查完告诉您具体情况。")],
    # 29
    [msg("user", "你好，我想报名参加这个周末的马拉松比赛。"),
     msg("assistant", "好的，这次比赛有全程、半程和迷你三个项目。您想参加哪个？"),
     msg("user", "我是第一次跑马拉松，选哪个比较合适？"),
     msg("assistant", "如果是第一次参加的话，建议选迷你马拉松，全程五公里，比较轻松。等有了经验再挑战半程和全程。")],
    # 30
    [msg("user", "你们这里能办护照吗？"),
     msg("assistant", "可以的。办护照需要您本人到场，带好身份证和户口本原件，还需要现场拍照。"),
     msg("user", "大概多久能拿到？"),
     msg("assistant", "一般是七个工作日。如果您有紧急出行需要，可以申请加急办理，三个工作日就能拿到，但需要额外的费用。")],
    # 31
    [msg("user", "你有没有读过什么好书推荐一下？"),
     msg("assistant", "最近在读一本关于心理学的书，叫《思考，快与慢》，非常有意思。它讲的是人类思维的两种模式，读完会对自己的决策方式有新的认识。"),
     msg("user", "听起来不错，在哪里能买到？"),
     msg("assistant", "网上书店都有卖的，电子版也有，很方便。如果你喜欢实体书的话，附近的书店应该也有。")],
    # 32
    [msg("user", "今天的股市行情怎么样？"),
     msg("assistant", "今天大盘整体表现平稳，上证指数微涨了零点三个百分点。新能源板块比较强势，消费板块有所回调。"),
     msg("user", "你觉得新能源还能继续涨吗？"),
     msg("assistant", "股市的走势很难预测。投资需要谨慎，建议你根据自己的风险承受能力来做决定，不要盲目追涨。")],
    # 33
    [msg("user", "我想学游泳，但是有点怕水。"),
     msg("assistant", "怕水是很正常的，很多人刚学的时候都会有这种感觉。建议你先在浅水区适应一下，慢慢把头浸入水中练习憋气。"),
     msg("user", "需要请教练吗？"),
     msg("assistant", "建议请一个教练，有专业指导的话学得更快也更安全。一般十节课左右就能掌握基本的蛙泳了。")],
    # 34
    [msg("user", "你好，我想问一下这个岗位的工作内容。"),
     msg("assistant", "这个岗位主要负责产品的日常运营和数据分析，同时需要跟技术团队配合推动产品迭代。"),
     msg("user", "需要加班吗？"),
     msg("assistant", "平时基本准时下班，但在项目上线前可能需要加班。公司有弹性工作制度，也会有加班补贴。")],
    # 35
    [msg("user", "请帮我推荐一款适合新手的相机。"),
     msg("assistant", "如果你是刚入门的话，可以考虑微单相机，体积小、操作简单，画质也不错。价格在三千到五千之间有很多不错的选择。"),
     msg("user", "微单和单反有什么区别？"),
     msg("assistant", "最大的区别是微单没有反光镜，所以体积更小更轻。现在微单的性能已经很强了，对于大多数人来说完全够用。")],
    # 36
    [msg("user", "你好，我预约了下午三点看牙。"),
     msg("assistant", "好的，请问您贵姓？"),
     msg("user", "我姓李。"),
     msg("assistant", "李先生您好，找到您的预约了。请先到候诊区稍等一下，医生大概五分钟后就能接诊。")],
    # 37
    [msg("user", "你觉得远程办公好还是去办公室上班好？"),
     msg("assistant", "两种方式各有优缺点。远程办公时间更灵活，不用通勤，但可能会影响团队协作。去办公室上班沟通更方便，但通勤时间是个问题。"),
     msg("user", "我们公司打算推行混合办公。"),
     msg("assistant", "混合办公是个不错的选择，兼顾了灵活性和协作需求。关键是要跟团队约定好哪些天必须到办公室，保证重要会议的面对面沟通。")],
    # 38
    [msg("user", "这个菜放了多少盐？也太咸了吧。"),
     msg("assistant", "不好意思，可能盐放多了。我给您重新做一份，好吗？"),
     msg("user", "好的，麻烦了。口味清淡一点。"),
     msg("assistant", "没问题，这次一定注意。稍等几分钟就好。")],
    # 39
    [msg("user", "你知道附近哪里有加油站吗？"),
     msg("assistant", "前面路口右转大概三百米就有一个中石化的加油站。"),
     msg("user", "那里价格怎么样？"),
     msg("assistant", "价格跟其他中石化的一样，都是统一定价的。那个加油站还有便利店和洗车服务，挺方便的。")],
    # 40
    [msg("user", "孩子发烧了，体温三十八度五，该怎么办？"),
     msg("assistant", "先不要着急。可以先给孩子物理降温，用温水擦拭额头和腋下，多喝水。如果体温超过三十九度或者持续不退，就需要及时去医院。"),
     msg("user", "能吃退烧药吗？"),
     msg("assistant", "三十八度五可以适当吃一些儿童退烧药，注意按照说明书的剂量。同时观察孩子的精神状态，如果出现嗜睡或者抽搐的情况要马上就医。")],
    # 41
    [msg("user", "你们的外卖大概多久能送到？"),
     msg("assistant", "一般情况下四十分钟左右。如果赶上用餐高峰期可能会稍慢一些。"),
     msg("user", "能快一点吗？我赶时间。"),
     msg("assistant", "我帮您备注加急，厨房会优先处理。不过配送时间也取决于骑手的路况，我们尽量为您加快。")],
    # 42
    [msg("user", "今天面试感觉怎么样？"),
     msg("assistant", "整体还行，面试官问的问题都回答上了。就是有一道技术题，我不太确定自己答得对不对。"),
     msg("user", "别想太多了，等结果就好。"),
     msg("assistant", "也是，反正也改变不了什么了。如果这次不行就继续投其他公司吧。")],
    # 43
    [msg("user", "服务员，麻烦结账。"),
     msg("assistant", "好的，请稍等。您这桌一共消费了三百二十八元。请问现金还是扫码支付？"),
     msg("user", "扫码吧。"),
     msg("assistant", "好的，这是我们的收款码。付款成功后我给您打印发票。")],
    # 44
    [msg("user", "下周一有个重要的汇报，我好紧张。"),
     msg("assistant", "紧张是正常的，说明你重视这件事。充分的准备是最好的定心丸，把汇报内容多练几遍，想好可能被问到的问题。"),
     msg("user", "如果领导问的问题我答不上来怎么办？"),
     msg("assistant", "诚实地说你会后续确认，比硬编一个答案要好得多。领导一般也能理解，不会要求你什么都知道。")],
    # 45
    [msg("user", "你觉得该不该跟室友谈一下卫生问题？"),
     msg("assistant", "如果影响到你的生活了，当然应该谈。态度要友好，就事论事，不要带情绪。比如可以提议大家一起制定一个值日表。"),
     msg("user", "我怕谈了之后关系变尴尬。"),
     msg("assistant", "换个角度想，如果一直不说，你心里的不满会越积越多，最后反而更容易爆发冲突。早点沟通，心平气和地聊，一般都能解决。")],
    # 46
    [msg("user", "我最近想换个发型，你有什么推荐？"),
     msg("assistant", "这要看你的脸型和日常风格。如果你的脸比较圆，可以试试有层次感的中长发。如果是长脸的话，刘海能起到修饰的效果。"),
     msg("user", "我想剪短发，会不会显得太成熟了？"),
     msg("assistant", "不会，短发也有很多年轻时尚的款式。你可以找一些喜欢的短发造型图片给理发师看，这样沟通起来更清楚。")],
    # 47
    [msg("user", "请问这个航班可以选座位吗？"),
     msg("assistant", "可以的。您可以在值机的时候选择靠窗或者靠过道的位置。提前在线值机的话，选择会更多。"),
     msg("user", "我想要靠窗的座位。"),
     msg("assistant", "好的，我帮您选了32A，靠窗的位置。登机牌已经打印好了，请拿好。祝您旅途愉快。")],
    # 48
    [msg("user", "你们店能修笔记本电脑吗？"),
     msg("assistant", "可以的。请问电脑是什么问题？"),
     msg("user", "开机特别慢，而且经常死机。"),
     msg("assistant", "这种情况可能是硬盘老化或者系统问题。我先帮您检测一下，如果是硬盘的话建议换一个固态硬盘，速度会快很多。")],
    # 49
    [msg("user", "明天的家长会几点开始？"),
     msg("assistant", "明天下午两点半在学校大礼堂开始。班主任会先介绍班级情况，然后是各科老师跟家长交流。"),
     msg("user", "大概多久结束？"),
     msg("assistant", "一般两个小时左右。结束后如果您有单独的问题想跟老师聊，也可以留下来。")],
    # 50
    [msg("user", "我打算国庆去云南旅游，有什么建议？"),
     msg("assistant", "云南很值得去！推荐大理、丽江和香格里拉这条经典路线。国庆期间是旺季，酒店和机票要提前订。"),
     msg("user", "需要带什么特别的东西吗？"),
     msg("assistant", "云南紫外线很强，一定要带好防晒霜和墨镜。昼夜温差大，带一件薄外套。如果要去香格里拉的话，海拔比较高，可以提前准备一些抗高反的药物。")],
]

# ============================================================
# 50 Chinese-English mixed conversations
# ============================================================
mix_conversations = [
    # 1
    [msg("user", "我想improve一下我的英语口语，有什么好的方法吗？"),
     msg("assistant", "最有效的方法就是多说多练。你可以找一个language exchange partner，每周practice几次。看英文电影的时候试着跟读，也很有帮助。")],
    # 2
    [msg("user", "今天的meeting几点开始？"),
     msg("assistant", "下午两点在conference room B，记得带上你的project proposal。"),
     msg("user", "好的，我需要准备presentation吗？"),
     msg("assistant", "不用，今天主要是brainstorm session，大家一起讨论就好。")],
    # 3
    [msg("user", "这个project的deadline是什么时候？"),
     msg("assistant", "原定是下周五，不过PM说可以extend到下下周一。你那边的progress怎么样了？"),
     msg("user", "Backend的部分基本done了，frontend还需要几天。"),
     msg("assistant", "那时间应该够。记得写好unit test，上次code review的时候lead特别强调了这一点。")],
    # 4
    [msg("user", "请问你们这里accept信用卡吗？"),
     msg("assistant", "Yes，我们接受所有主流credit card，Visa、MasterCard和银联都可以。也支持mobile payment，微信和支付宝都行。")],
    # 5
    [msg("user", "我刚拿到一个job offer，但是不确定要不要accept。"),
     msg("assistant", "恭喜你！可以从几个方面考虑：salary和benefits满意吗？公司的culture和发展前景怎么样？跟你的career plan是否match？"),
     msg("user", "Salary比现在高30%，但是要relocate到另一个城市。"),
     msg("assistant", "这确实是个tough decision。建议你列一个pros and cons的清单，也跟家人好好discuss一下。搬到新城市既是挑战也是机会。")],
    # 6
    [msg("user", "你觉得AI会不会replace很多工作？"),
     msg("assistant", "AI确实会automate一些重复性的工作，但同时也会create很多新的岗位。关键是要keep learning，提升自己不容易被替代的skills。"),
     msg("user", "比如哪些skills比较重要？"),
     msg("assistant", "Critical thinking、creativity和interpersonal skills这些是AI比较难替代的。同时如果你能学会跟AI协作，反而会成为你的competitive advantage。")],
    # 7
    [msg("user", "昨天看了一个很好的TED talk，关于time management的。"),
     msg("assistant", "是吗？讲了什么内容？我最近也觉得time management是个大问题。"),
     msg("user", "Speaker说最重要的是学会say no，把energy放在most important的事情上。"),
     msg("assistant", "这个观点我很agree。很多时候我们busy但不productive，就是因为没有prioritize好。")],
    # 8
    [msg("user", "Help me看一下这段code有什么问题。"),
     msg("assistant", "我看了一下，主要有两个issue。第一个是你的for loop里有一个off-by-one error，应该是小于length而不是小于等于。第二个是这个variable没有initialize就直接用了。"),
     msg("user", "啊，怪不得一直报error。"),
     msg("assistant", "改完之后记得run一下test cases。另外建议你加一些error handling，防止edge cases出问题。")],
    # 9
    [msg("user", "这个weekend你有什么plan？"),
     msg("assistant", "我打算去hiking，听说郊外有一条新的trail风景特别好。你要不要一起来？"),
     msg("user", "听起来不错！需要带什么equipment吗？"),
     msg("assistant", "穿一双comfortable的hiking shoes就行，然后带够water和一些snacks。路程不长，大概三个小时能走完。")],
    # 10
    [msg("user", "我最近在考虑是否要读一个MBA。"),
     msg("assistant", "MBA是一个不小的investment，不管是时间还是money。你的motivation是什么？是想career switch还是想升到management level？"),
     msg("user", "主要是想往management方向发展。"),
     msg("assistant", "如果是这样的话，MBA确实有帮助。不过也可以考虑part-time的program，这样不用quit你现在的工作。很多top schools都有weekend或者evening的选项。")],
    # 11
    [msg("user", "我的laptop突然blue screen了，怎么办？"),
     msg("assistant", "先不要panic。你记得blue screen上显示的error code吗？不同的code代表不同的问题。"),
     msg("user", "好像是什么memory相关的。"),
     msg("assistant", "那可能是RAM有问题。你可以先restart看看能不能正常开机。如果反复出现，建议去check一下memory，可能需要replace了。")],
    # 12
    [msg("user", "今天的lunch你想吃什么？"),
     msg("assistant", "我都行，你有什么preference吗？"),
     msg("user", "新开了一家Japanese restaurant，reviews很不错，要不要去try一下？"),
     msg("assistant", "好啊！我正好很久没吃sashimi了。走吧，早点去免得要排队。")],
    # 13
    [msg("user", "最近在学Python，感觉data structure那部分好难。"),
     msg("assistant", "刚开始是有点challenging。建议你先把list、dictionary和set这些basic的搞清楚，然后再学stack和queue这些。"),
     msg("user", "有什么好的learning resource推荐吗？"),
     msg("assistant", "LeetCode上有很多练习题，从easy开始做。另外推荐一本书叫《算法图解》，讲得很visual，很容易understand。")],
    # 14
    [msg("user", "你对remote work怎么看？"),
     msg("assistant", "我觉得remote work最大的benefit就是flexibility，省去了commute的时间。但也有challenge，比如容易分心，work-life balance不好把握。"),
     msg("user", "你们team是怎么handle的？"),
     msg("assistant", "我们每天有一个15分钟的standup meeting保持sync，然后用Slack来日常communicate。重要的discussion还是会安排video call。")],
    # 15
    [msg("user", "这个bug搞了一整天还是没fix。"),
     msg("assistant", "什么样的bug？describe一下symptom。"),
     msg("user", "API call有时候会return 500 error，但不是每次都这样，很random。"),
     msg("assistant", "听起来像是race condition或者timeout的问题。你check一下server log，看看error出现的时候backend那边具体是什么exception。可能是database connection pool的问题。")],
    # 16
    [msg("user", "我打算start一个side project，做一个fitness tracking的app。"),
     msg("assistant", "很好的idea！你想用什么tech stack？"),
     msg("user", "前端用React Native，backend用Node.js。"),
     msg("assistant", "不错的选择，cross-platform又能share code。建议你先做一个MVP，core features先跑起来，然后再iterate。别一开始就想做得太完美。")],
    # 17
    [msg("user", "我的resume需要update一下，你能给点advice吗？"),
     msg("assistant", "当然。首先确保你的work experience是按时间倒序排列的，最近的在最上面。每段经历用bullet points描述，尽量quantify你的achievements，比如'提升了30% performance'。"),
     msg("user", "需要写多长比较合适？"),
     msg("assistant", "如果你的工作经验在十年以内，一页纸就够了。Keep it concise，recruiter平均花六秒scan一份resume，所以关键信息要一目了然。")],
    # 18
    [msg("user", "今天的standup你说的那个blocker解决了吗？"),
     msg("assistant", "解决了，是dependency的version conflict。我upgrade了之后重新build就pass了。"),
     msg("user", "那sprint里剩下的task应该能on track完成吧？"),
     msg("assistant", "应该没问题，我今天下午把最后一个feature branch merge进去，明天做final testing。")],
    # 19
    [msg("user", "公司要organize一个team building，你有什么好的suggestion？"),
     msg("assistant", "可以考虑escape room，既好玩又能锻炼teamwork。或者做一次outdoor barbecue，大家放松一下。"),
     msg("user", "Escape room听起来很fun！大概需要多少budget？"),
     msg("assistant", "一般人均一百到两百左右。如果team比较大的话，可以分成几个小组compete，更有意思。")],
    # 20
    [msg("user", "我觉得我的English writing需要improve，你有什么建议？"),
     msg("assistant", "多读是最好的方法。推荐你看一些quality的英文文章，比如The Economist或者New York Times。另外写完之后可以用Grammarly check一下grammar。"),
     msg("user", "平时工作中写email总是斟酌很久。"),
     msg("assistant", "Business email其实有很多template可以参考。Keep it short and clear，先说purpose，再给detail，最后写action item。多写几次就会越来越熟练。")],
    # 21
    [msg("user", "昨天interview了一个candidate，感觉technical skills很strong。"),
     msg("assistant", "那culture fit怎么样？我们team比较注重collaboration。"),
     msg("user", "Communication能力也不错，回答问题很structured。"),
     msg("assistant", "那可以进入next round。我这边安排一个system design的面试，再evaluate一下他的big picture thinking。")],
    # 22
    [msg("user", "你们的product支持API integration吗？"),
     msg("assistant", "支持的，我们提供RESTful API和webhook两种方式。Documentation在developer portal上可以查到。"),
     msg("user", "有没有SDK？我们主要用Java。"),
     msg("assistant", "有的，我们提供Java、Python和Node.js的official SDK。GitHub上都能找到，里面也有sample code可以参考。")],
    # 23
    [msg("user", "这次performance review你打算怎么prepare？"),
     msg("assistant", "我把这半年的key achievements都整理了一下，还有一些metrics来quantify我的contribution。"),
     msg("user", "记得也想好你下半年的goal。"),
     msg("assistant", "对，我打算focus在leadership skills的提升上，想take on更多cross-team的project。")],
    # 24
    [msg("user", "Design那边说我们的UX需要redesign。"),
     msg("assistant", "哪些部分需要改？是user feedback driven的还是他们自己的想法？"),
     msg("user", "主要是onboarding flow，用户的drop-off rate比较高。"),
     msg("assistant", "那确实需要optimize。建议先做一些user research，了解清楚pain points在哪里，然后再redesign。不要光靠assumption。")],
    # 25
    [msg("user", "我想给家里装smart home设备，你有什么recommend？"),
     msg("assistant", "可以先从smart speaker开始，比如小爱同学或者天猫精灵，然后逐步添加smart light和smart lock。"),
     msg("user", "不同brand的设备能compatible吗？"),
     msg("assistant", "现在很多设备都support Matter protocol了，cross-brand的compatibility比以前好多了。买之前check一下specifications，确保都是同一个ecosystem或者支持Matter。")],
    # 26
    [msg("user", "我们team要migrate到新的cloud platform，你觉得AWS还是Azure比较好？"),
     msg("assistant", "这要看你们的具体needs。AWS的service最全，community也最大。Azure跟Microsoft的生态integrate得好，如果你们用Office 365的话会比较方便。"),
     msg("user", "我们现在用的是Office 365。"),
     msg("assistant", "那Azure可能是更natural的选择。而且Azure对enterprise的support做得不错，pricing跟你们现有的Microsoft license可能还能bundle。建议先做一个pilot project试试。")],
    # 27
    [msg("user", "Hey，下班后要不要去gym？"),
     msg("assistant", "好啊！今天刚好是leg day，一起练。"),
     msg("user", "我最近想focus on core training，有什么好的exercise推荐吗？"),
     msg("assistant", "Plank是最basic也最effective的。还有Russian twist和dead bug也很好。每个动作做三组，每组做到failure就行。")],
    # 28
    [msg("user", "客户那边要求我们的system做security audit。"),
     msg("assistant", "这是standard的要求。我们需要先做一个internal assessment，check一下有没有obvious的vulnerability。"),
     msg("user", "Authentication那部分我比较concern。"),
     msg("assistant", "我建议请一个third-party的security firm来做penetration testing。另外确保我们的encryption标准符合industry best practice，data at rest和in transit都要cover到。")],
    # 29
    [msg("user", "你觉得ChatGPT和Claude哪个更好用？"),
     msg("assistant", "各有strengths。Depends on你的use case。写code的话两个都不错，长文本analysis的话Claude的context window更大。最好的方法是都try一下，看哪个更suit你的需求。")],
    # 30
    [msg("user", "这个feature request的priority是什么？"),
     msg("assistant", "PM定的是P1，因为好几个big client都在要这个feature。"),
     msg("user", "那我们这个sprint就要start了？"),
     msg("assistant", "是的，已经在backlog里了。我建议先花两天时间做technical design，然后review一下再开始coding。这个feature涉及到的component比较多，要plan好。")],
    # 31
    [msg("user", "最近crypto market波动好大。"),
     msg("assistant", "是的，Bitcoin前几天又跌了不少。Crypto本身volatility就很高，invest的话一定要用你能承受lose的钱。"),
     msg("user", "你觉得长期来看怎么样？"),
     msg("assistant", "长期趋势很难predict。Blockchain technology本身是有价值的，但具体哪个token能长期survive就不好说了。建议diversify你的portfolio，不要all in任何一个asset。")],
    # 32
    [msg("user", "Marketing team想做一个新的campaign，需要我们提供一些data。"),
     msg("assistant", "什么类型的data？User demographics还是usage analytics？"),
     msg("user", "主要是user engagement的数据，比如DAU、retention rate这些。"),
     msg("assistant", "这些在我们的analytics dashboard上都有。我下午export一份report给他们。需要注意data privacy的问题，确保不要包含any personally identifiable information。")],
    # 33
    [msg("user", "我的visa快要expire了，需要renew。"),
     msg("assistant", "你是什么type的visa？不同类型的renew process不一样。"),
     msg("user", "工作签证，H1B。"),
     msg("assistant", "H1B的renewal需要你employer file一个extension petition。建议提前六个月开始prepare，因为processing time比较长。跟你公司的HR和immigration lawyer尽快联系。")],
    # 34
    [msg("user", "我们team需要recruit两个engineer，你觉得job description怎么写？"),
     msg("assistant", "首先明确role的level和responsibilities，然后列出required skills和nice-to-have skills。Don't make the requirements太long，否则会discourage很多qualified的candidate申请。"),
     msg("user", "Tech stack需要写得很detail吗？"),
     msg("assistant", "写主要的就行。比如我们用React和Go，就写清楚。具体到library level的就不用了，有solid基础的人都能快速pick up。")],
    # 35
    [msg("user", "你听说了吗？公司要implement一个新的OKR system。"),
     msg("assistant", "听说了。HR那边在做training session，下周所有manager都要参加。"),
     msg("user", "OKR跟我们之前用的KPI有什么区别？"),
     msg("assistant", "OKR更focus on outcomes而不是outputs。每个objective下面有几个measurable的key results。OKR鼓励set ambitious的目标，完成70%就算不错了，跟KPI要求100%达标的思路不一样。")],
    # 36
    [msg("user", "我在写一篇blog post，关于microservices vs monolith的。"),
     msg("assistant", "很好的topic！你打算从哪个angle来写？"),
     msg("user", "想结合我们team从monolith迁移到microservices的experience。"),
     msg("assistant", "那非常有real-world的value。记得cover一下migration过程中遇到的challenge，比如service之间的communication、data consistency这些。读者最感兴趣的往往是lessons learned而不是theory。")],
    # 37
    [msg("user", "新来的intern表现怎么样？"),
     msg("assistant", "学习能力很strong，上手很快。不过还需要在code quality方面多指导一下，commit message和documentation写得不太规范。"),
     msg("user", "是刚毕业的嘛，正常的。"),
     msg("assistant", "对，我给他assign了一个mentor，每周做一次one-on-one。再过一个月应该就能independently handle一些task了。")],
    # 38
    [msg("user", "这周的release plan确定了吗？"),
     msg("assistant", "计划周四做staging deploy，周五observe一天，如果没有major issue的话下周一push到production。"),
     msg("user", "Rollback plan准备好了吗？"),
     msg("assistant", "准备好了，database migration有backward compatibility，所以rollback的话直接switch回previous version就行。我会在release document里详细写清楚steps。")],
    # 39
    [msg("user", "你的MacBook Pro用了几年了？准备upgrade吗？"),
     msg("assistant", "快三年了。Performance还行，就是battery有点不行了。在考虑要不要换最新的M系列chip的。"),
     msg("user", "新款确实performance提升很大。"),
     msg("assistant", "是的，特别是跑local的machine learning model，速度快了很多。不过也要看公司有没有budget，不然就自己掏腰包了。")],
    # 40
    [msg("user", "你有没有用过Notion？我想找一个好的note-taking tool。"),
     msg("assistant", "用过，Notion很versatile，可以做笔记、project management、wiki什么的。不过如果你只需要simple的note-taking，Obsidian可能更lightweight。"),
     msg("user", "我主要是想manage工作中的各种information。"),
     msg("assistant", "那Notion更适合你，它的database功能很powerful，可以create不同的view来organize你的信息。Team collaboration也做得不错。")],
    # 41
    [msg("user", "Client那边complain说我们的app loading太慢了。"),
     msg("assistant", "需要profile一下看bottleneck在哪里。是initial load慢还是navigation慢？"),
     msg("user", "主要是initial load，第一次打开要五六秒。"),
     msg("assistant", "那大概率是bundle size太大了。可以做code splitting和lazy loading来optimize。另外check一下是不是有什么heavy的dependency可以替换掉或者tree-shake。")],
    # 42
    [msg("user", "这个weekend要参加一个hackathon，你们team做什么project？"),
     msg("assistant", "我们打算做一个AI-powered的meal planning tool，根据用户的dietary preferences和冰箱里的食材来recommend recipes。"),
     msg("user", "这个idea很cool！打算用什么model？"),
     msg("assistant", "Plan是用一个open source的LLM做recipe generation，前端用Next.js快速搭。Hackathon嘛，focus on demo effect，不用做得太production-ready。")],
    # 43
    [msg("user", "我想在LinkedIn上build一下personal brand。"),
     msg("assistant", "First step是完善你的profile，写一个compelling的headline和summary。然后定期post一些你的professional insights或者industry相关的内容。"),
     msg("user", "需要每天post吗？"),
     msg("assistant", "不用，quality比quantity重要。每周两三次就够了。Engage with别人的post也很重要，comment和share都能增加你的visibility。")],
    # 44
    [msg("user", "今天的workshop你觉得有收获吗？"),
     msg("assistant", "还不错，关于agile的那部分讲得很practical。特别是关于如何做retrospective的那个framework很useful。"),
     msg("user", "我觉得关于estimation那部分也很有启发。"),
     msg("assistant", "对，用story points来estimate确实比用小时数更make sense。避免了over-commitment的问题。我打算下个sprint就试试。")],
    # 45
    [msg("user", "今年的annual leave你打算怎么安排？"),
     msg("assistant", "打算年底take两周假去Japan旅游。你呢？"),
     msg("user", "我在考虑去Thailand或者Bali。"),
     msg("assistant", "两个地方都很nice。Thailand的food超级好吃，cost也比较低。Bali比较chill，适合relax。看你prefer adventure还是relaxation了。")],
    # 46
    [msg("user", "Database的query越来越slow了，需要optimize一下。"),
     msg("assistant", "先check一下slow query log，看哪些query最consume resources。"),
     msg("user", "我怀疑是index没有建好。"),
     msg("assistant", "很有可能。你用EXPLAIN analyze一下那几个慢query，看是不是在做full table scan。该加的index加上，同时注意不要over-index，否则write performance会受影响。")],
    # 47
    [msg("user", "我在犹豫要不要学Rust。"),
     msg("assistant", "Rust的learning curve比较steep，但一旦学会了，写出来的代码performance和safety都很好。你现在主要用什么language？"),
     msg("user", "主要是Python和Go。"),
     msg("assistant", "如果你需要做systems programming或者对performance要求很高的项目，Rust是个great choice。不过如果现有的tech stack能满足需求，也不用为了学而学。看你的career发展方向。")],
    # 48
    [msg("user", "我们的CI/CD pipeline经常fail，影响dev velocity。"),
     msg("assistant", "Flaky test是最常见的原因。先统计一下哪些test fail rate最高，然后集中fix掉。"),
     msg("user", "还有就是build时间太长。"),
     msg("assistant", "可以看看能不能parallelize一些steps，或者implement caching来避免重复下载dependencies。Docker layer caching如果没有用的话也建议加上，能省很多时间。")],
    # 49
    [msg("user", "你平时怎么keep up with最新的tech trends？"),
     msg("assistant", "我主要是看Hacker News和一些tech blog，还subscribe了几个newsletter，比如TLDR和ByteByteGo。Podcast也会听，commute的时候很方便。"),
     msg("user", "中文的tech content呢？"),
     msg("assistant", "InfoQ中文站不错，还有掘金社区。另外B站上有很多quality很高的technical talk和tutorial。")],
    # 50
    [msg("user", "我们的documentation太outdated了，需要overhaul一下。"),
     msg("assistant", "同意，好几个API的doc跟actual behavior都不match了。建议我们先prioritize最常用的部分。"),
     msg("user", "谁来负责这个work？"),
     msg("assistant", "可以每个team member负责自己most familiar的module。设一个deadline，比如两周内全部update完。另外建议在CI里加一个check，每次code change都要同步update对应的documentation。")],
]

# ============================================================
# Write to JSONL
# ============================================================
output_file = "./zh_mixed_samples.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for messages in zh_conversations:
        line = json.dumps(conv(SYSTEM_ZH, messages), ensure_ascii=False)
        f.write(line + "\n")
    for messages in mix_conversations:
        line = json.dumps(conv(SYSTEM_MIX, messages), ensure_ascii=False)
        f.write(line + "\n")

print(f"Done. Wrote {len(zh_conversations)} Chinese + {len(mix_conversations)} mixed samples to {output_file}")
