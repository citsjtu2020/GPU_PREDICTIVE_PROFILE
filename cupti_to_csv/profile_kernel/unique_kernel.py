import pandas as pd

# 读取CSV文件
df = pd.read_csv(' metric_results_allsess.csv')

# 删除基于'kernel'列的重复行，保持第一个出现的条目
df_unique = df.drop_duplicates(subset='kernel', keep='first')
#df_unique = df.drop_duplicates(subset='name', keep='first')
# 将结果保存回新的CSV文件
df_unique.to_csv('unique_kernels_1.csv', index=False)


# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('metric_results_allsess.csv')

# # 删除基于'kernel'和'value'列的重复行，保持第一个出现的条目
# df_unique = df.drop_duplicates(subset=['kernel', 'value'], keep='first')

# #df_sorted = df_unique.sort_values(by='kernel')
# # 将结果保存回新的CSV文件
# df_unique.to_csv('unique_kernels_with_values.csv', index=False)


