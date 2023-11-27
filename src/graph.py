from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.faker import Faker

print(Faker.choose())
print( Faker.values())
c = (
    Scatter()
    .add_xaxis( Faker.values())
    .add_yaxis("Y", Faker.values())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="SOM二維分群圖"),
        xaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
    )
    .render("scatter_splitline.html")
)