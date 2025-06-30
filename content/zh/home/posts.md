+++
# 使用Pages小部件创建的最近博客文章部分。
# 此部分显示来自`content/zh/post/`的最近博客文章。

widget = "pages"  # 参见 https://sourcethemes.com/academic/docs/page-builder/
headless = true  # 这个文件代表一个页面部分。
active = true  # 激活这个小部件？true/false
weight = 15  # 此部分将出现的顺序。

title = "最近文章"
subtitle = ""

[content]
  # 要显示的页面类型。例如：post, talk, 或 publication。
  page_type = "post"
  
  # 选择要显示的页面数量（0 = 所有页面）
  count = 10
  
  # 选择要偏移的页面数量
  offset = 0

  # 页面顺序。降序（desc）或升序（asc）日期。
  order = "desc"

  # 按分类术语过滤文章。
  [content.filters]
    tag = ""
    category = ""
    publication_type = ""
    exclude_featured = false
  
[design]
  # 在各种页面布局类型之间切换。
  #   1 = 列表
  #   2 = 紧凑
  #   3 = 卡片
  #   4 = 引用（仅限出版物）
  view = 2
  
[design.background]
  # 应用背景颜色、渐变或图像。
  #   取消注释（通过删除`#`）一个选项来应用它。
  #   通过设置`text_color_light`选择浅色或深色文本颜色。
  #   任何HTML颜色名称或十六进制值都有效。
  
  # 背景颜色。
  # color = "navy"
  
  # 背景渐变。
  # gradient_start = "DeepSkyBlue"
  # gradient_end = "SkyBlue"
  
  # 背景图像。
  # image = "background.jpg"  # `static/img/`中的图像名称。
  # image_darken = 0.6  # 使图像变暗？范围0-1，其中0是透明的，1是不透明的。

  # 文本颜色（true=浅色或false=深色）。
  # text_color_light = true
  
[advanced]
 # 自定义CSS。
 css_style = ""
 
 # CSS类。
 css_class = ""
+++
