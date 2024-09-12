scale_legend='nnz(P)+nnz(A)'
stack_legends=['Factor', 'SpMV', 'Solve', 'Vector']
# app_list=['Eq QP', 'SVM', 'Control', 'Portfolio', 'Huber', 'Lasso']
app_list=['SVM', 'Control', 'Portfolio', 'Huber', 'Lasso']
blue_code='#4862b9'
green_code='#63a645'
yellow_code='#f5c342'
color_list = [green_code, blue_code, yellow_code, 'purple'] 

def df_insert_row(df, row):
	df.loc[len(df)] = row

def get_app_code(app_name):
	return app_name.lower().replace(" ", "")
