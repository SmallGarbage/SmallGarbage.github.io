### Excel操作

#### 读Excel

``` python
import xlrd
# 打开一个Excel
work_book = xlrd.open_workbook(file_paht)
# 读取里面的一个sheet页
table = work_book.sheets()[n]  //n代表第几个sheet页
# 取sheet页中行数据
table.row_values(i)[0:]   // 取第i行，全部数据
# 取sheet页中列数据
table.col_values(i)[0:]   // 取第i列，全部数据
```

#### 写Excel

```python
import xlwt

# 创建一个Excel
work_book = xlwt.Workbook(encoding = 'utf-8')
# 新建一个sheet页
sheet = work_book.add_sheet("sheet1")
# 向sheet页中写入数据
sheet.write(0, 1, "string") // 向第0行第1列写入 string
```

