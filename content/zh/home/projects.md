+++
# 使用Portfolio小部件创建的项目部分。
widget = "portfolio"  # 参见 https://sourcethemes.com/academic/docs/page-builder/
headless = true  # 这个文件代表一个页面部分。
active = true  # 激活这个小部件？true/false
weight = 25  # 此部分将出现的顺序。

title = "项目"
subtitle = ""

[content]
  # 要显示的页面类型。例如：project。
  page_type = "project"
  
  # 过滤工具栏（可选）。
  # 根据需要添加或删除任意数量的过滤器（`[[content.filter_button]]`实例）。
  # 要显示所有项目，将`tag`设置为"*"。
  # 要按特定标签过滤，将`tag`设置为现有标签名称。
  # 要删除工具栏，删除/注释下面所有的`[[content.filter_button]]`实例。
  
  # 默认过滤器索引（例如，0对应于下面的第一个`[[filter_button]]`实例）。
  filter_default = 0
  
  # [[content.filter_button]]
  #   name = "全部"
  #   tag = "*"
  
  # [[content.filter_button]]
  #   name = "深度学习"
  #   tag = "Deep Learning"
  
  # [[content.filter_button]]
  #   name = "其他"
  #   tag = "Demo"

[design]
  # 选择此部分有多少列。有效值：1或2。
  columns = "2"

  # 在各种页面布局类型之间切换。
  #   1 = 列表
  #   2 = 紧凑
  #   3 = 卡片
  #   5 = 展示
  view = 3

  # 对于展示视图，是否翻转交替行？
  flip_alt_rows = false

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

