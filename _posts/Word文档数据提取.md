### Word数据提取

**原始数据**：*Word文档（各个项目点模板差距有点大）这里以标准文档为例图片经过打码处理*

![example](https://github.com/SmallGarbage/SmallGarbage.github.io/blob/master/example.png)

提取方法：将Word文档中所有内容转化成字符串，利用正则表达式实现待提取字段匹配

项目结构：

![image1](https://github.com/SmallGarbage/SmallGarbage.github.io/blob/master/img/image1.png)

定义方法类 初始化需要提取字段初始值

![image2](https://github.com/SmallGarbage/SmallGarbage.github.io/blob/master/img/image2.png)

图片代码中的 

```python
def __init__(self.file_path,word,finish, finish_txt) # finish， finish_txt 可不写下文完全没用到，懒得删
```

定义好了各个字段的初始值目的是为了 当有匹配不到的字段出现时，我们就用这个初始值替代，为此，我们需要写个小方法

```python
@staticmethod                      # 静态方法，类的工具包，放在函数前，该函数不传入self
def fill(para, value):
    if value is None or value == "":
        return para
    else:
        rerurn value
```

原始文件包括doc,docx,xls,xlsx,pdf  所以写了三个类型的文件处理方法，这就要先分下文件种类，在传入对应的方法处理，code如下：

```python
extension = file_path.split(".")[-1]            # 截取文档路径按“。“分割后文件后缀
if extension.lower() in ["doc", "docx"]:        # 判断 文件类型  并进行相应的处理
    self.file_type = "word"
    self.load_doc()
elif extension.lower() in ["xls", "xlsx"]:
	self.file_type = "excel"
	self.load_xls()
elif extension.lower() in ["pdf"]:
    self.file_type = "pdf"
    self.load_pdf()
else:
    warn_msg = "不知道你这是个啥格式文件"
    warning.warn(warn_msg)
```

先上一个处理doc的，也是我主要干的，Excel听说处理简单，那我怎么能干呢。当然是谁负责搞谁了，PDF太复杂，我选择无视。

处理文档分三步

1. 打开冰箱门（打开文档）
2. 把大象装进去（将文档内容简单处理并正则匹配）
3. 关上冰箱门（关闭文档，提裤子走人）

我们先进行第一步：

```python
from win32com.client imort Dispatch      # 先导一个win32com的包包，用这个包里的Dispath方法打开文档
def load_doc(self,word):
    word = Dispatch("Word.Application")  # 要是用WPS 可以手动更改Dispatch()中参数
    f = word.Documents.Open(self.file_path) # hhh 文档就这样被打开了
# 但是我想看一下可不可以print(f),发现不可以，还得干一件大事，开的不够彻底，在开
	comment = f.Content.Text    #  这样就可以了么 ，将f 转出文本形式就可以打印了，还不行
    comment = f.Content.Text.replace("\r", "")  # 将字符串里的回车 替换成空字符 就可以啦
# 然后为了后面正则匹配过程可以正常进行，我们将文档里的 空格，可能存在的特殊字符先统一处理一下下
   	comment = f.Content.Text.replace("\t", "").replace("＝", "=").replace("\r", "").replace("\xa0", "").replace(
            "\x07", "").replace("官腔", "管腔").replace("\u3000", "").replace("\x00", "").replace("\x01", "").replace(
            "\x15", "").replace("\x0c", "").replace("\x0e", "").replace("\x0c", "").replace("\x0b", "").replace(" ","").replace(
            ":", "").replace("：", "").replace("端", "段").replace("-", "")
# 然后这回 我们在  print(comment)  就很棒
    
```

然后就开始了我们苦逼的匹配之旅，主要是各个项目点模板不同，模板不同也就算了 还有错别字，错别字也就算了，还有填错表格了（填的内容将模板修改了），这也就算了 还有往里插入图片的，没有想不到的，我这里就举个例子看下匹配字段的过程。

```python
try:
    self.name = self.fill(self.name, re.search("姓名(.*?)受检者ID", comment).group(1))
except Exception:
    copyfile(self.file_path, "../output/err/{}".self.file_path("\\")[-1])
# 这种是利用正则表达式匹配字符串 还有一种是利用表格 看下面   
```

```python
for t in f.Tables:
    try:
        self.doctor = slef.fill(self.doctor, t.Cell(20, 6).Range.Text)
    except Exception:
        print("没匹配到")                  # 有一些文档，报告医生是按格子可以匹配出来的，但是有一些文档有点老版本，不行
```

写到这，突然发现有个最重要的忘写了，这个文档分左右侧，就是这个文档的内筒是上下两部分也就是所说的左右两侧，要分两侧的，思路如下：

```python
if "LCCA-IMT" in content:
    try:
        if content.index("LCCA-IMT") < content.index("RCCA-IMT"):
            left = re.search(LCCA-IMT(.*?)RCCA-IMT", content).group(1)
            right = re.search("RCCA-IMT(.*?)超声印象", content).group(1)
        else:
            right = re.search("RCCA-IMT(.*?)LCCA-IMT", content).group(1)
            left = re.search("LCCA-IMT(.*?)超声印象", content).group(1)
     except Exception:
            print("没匹配到")
else：
     try：
        if "右侧" in content and "左侧" in content:
             if content.index("左侧") < content.index("右侧"):
             	left = re.search("左侧(.*?)右侧", content).group(1)
                right = re.search("右侧(.*?)(超声印象|检查医生|报告医生)", content).group(1)
              else:
                   left = re.search("右侧(.*?)左侧", content).group(1)
                   right = re.search("左侧(.*?)(超声印象|检查提示)", content).group(1)
         else:
             try:
                left = re.search("右侧(.*?)右侧", content).group(1)
                content = re.sub(left, "", content)
                right = re.search("右侧(.*?)(超声印象|检查提示|报告医生)", content).group(1)
              except Exception:
                  try:
                     left = re.search("左侧(.*?)左侧", content).group(1)
                     content = re.sub(left, "", content)
                     right = re.search("左侧(.*?)(超声印象|检查提示|报告医生)", content).group(1)
                  except Exception:
                      try:
                          left = re.search("CCA-IMT（mm）(.*?)CCA-IMT（mm）", content).group(1)
                           content = re.sub(left, "", content)
                           right = re.search("CCA-IMT（mm）(.*?)(超声印象|检查医生|报告医生)", content).group(1)
                       except Exception:
                            copyfile(self.file_path,
                                             "../../../output/errr/xindong/{}".format(self.file_path.split("\\")[-1]))
                                    print("left or right", self.file_path)
                                    return 0
    except Exception:
       copyfile(self.file_path, "../../../output/errr/xindong/{}".format(self.file_path.split("\\")[-1]))
   	print("left or right", self.file_path)                  
```

