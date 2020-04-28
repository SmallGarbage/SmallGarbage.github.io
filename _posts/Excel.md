### xlrd :python操作Excel读Excel，xlwt是写Excel的库



1. #### 打开Excel文件，读取数据

```python
data = xlrd.open_workbook(filename)
```

- ##### 获取book中的一个表

```python
# 通过索引顺序获取
table = data.sheets()[0]

#通过索引顺序获取
table = data.sheet_by_index(sheet_index)

##通过名称获取
table = data.sheet_by_name(sheet_name)
```

- ##### 行的操作

```python
# 获取工作表中行数
nrows = table.nrows
```

- ##### 单元格的操作

```python
# 返回单元格对象
table.cell(rowx,colx)
```

1. ### xlwt写入Excel操作

```python
# 打开一个worksheet
workbook = xlwt.Workbbok(encoding='utf-8')
# 创建一个worksheet
worksheet = workbook.add_sheet('my worksheet')
# 写入Excel
worksheet.write(1,0,label="this is text")
workbook.save
```

