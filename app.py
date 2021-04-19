#!/usr/bin/env python
# coding: utf-8

# # Business Case #3 - Market Basket Analysis

# ## Authors:
# #### Débora Santos (m20200748),Pedro Henrique Medeiros (m20200742), Rebeca Pinheiro (m20201096)
# 
# #### Group D - D4B Consulting

# ### Installing and import packages
# 
# Maybe it will be necessary install some 'special' packages to this notebook works. 
# 
# Please follow the next cells and check if it's necessary

# In[1]:


#Install package mlxtend
#!pip install mlxtend


# In[2]:


# Import packages
import pandas as pd
import numpy as np
import datetime as dt
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px


# ### Collect initial data

# In[3]:


#import dataset
path = 'https://raw.githubusercontent.com/Debs86/Business_Cases_Projects/master/BC3/'
orders = pd.read_csv(path +'orders.csv',sep=",")
orders_prod = pd.read_csv(path +'order_products.csv',sep=",")
departments = pd.read_csv(path +'departments.csv',sep=",")
products = pd.read_csv(path +'products.csv',sep=",")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ### Creating the dataframes to plot the customers' behaviors

# In[4]:


#Create a function to return the number of orders by day of week or per hour of the day to be use in the app
def frequent_time(option):
    df = orders
    if option == "Day of week":
        freq = df['order_dow'].value_counts()
    elif option == 'Hour of the day':
        freq = df['order_hour_of_day'].value_counts()
    return freq


# In[5]:


#Create the dataframe grouped by days since prior order
prior_orders = orders['days_since_prior_order'].value_counts()


# In[6]:


#Create a dataframe grouped by user id
number_orders = orders['user_id'].value_counts()
#Sort the values by the user with highest number of orders
number_orders = number_orders.sort_values(ascending=False,axis=0)
#Reset index
number_orders = number_orders.reset_index()
#Rename columns
number_orders.rename(columns={'index': 'number_user'}, inplace = True)
number_orders.rename(columns={'user_id': 'number_orders'}, inplace = True)
#Group the dataframe by quantity of users according by number of orders placed
number_orders= number_orders.groupby("number_orders")['number_user'].agg(['count'])
#Reset index
number_orders = number_orders.reset_index()


# In[7]:


#Create a dataframe to plot the basket size in the dash
#Create the dataframe grouped by number of itens add to cart
avg_basket = orders_prod.groupby("order_id")['add_to_cart_order'].agg(['count'])
#Rename the column
avg_basket = avg_basket.rename({"count":"basket_size"}, axis=1)
#Reset the index the column
avg_basket.reset_index(inplace = True)
#Create the dataframe grouped by number of itens add to cart
avg_basket= avg_basket.groupby('basket_size')["order_id"].agg(['count'])
#Rename the column
avg_basket = avg_basket.rename({"count":"number_of_orders"}, axis=1)
#Reset the index the column
avg_basket.reset_index(inplace = True)
#Sort by number of orders according with the basket size.
avg_basket.sort_values(by='number_of_orders', ascending=False, inplace=True)
#Filter the 25 basket sizes that most appear. 
avg_basket_top25 = avg_basket[0:25]


# ### Creating the dataframes to plot the Products and departments most consumed

# In[8]:


#Create a dataframe to merge the datasets by product and department
df_merge = pd.merge(orders_prod, products, how='left', on='product_id')
df_merge = pd.merge(df_merge, departments, how='left', on='department_id')


# In[9]:


#Products more frequent by itens sold
prod_freq = df_merge['product_name'].value_counts()


# In[10]:


# Create dataframe to plot the number of orders by products

#Drop the columns not useful
df1 = df_merge.drop(["product_id", "department_id", 'add_to_cart_order','reordered','department'], axis=1)
#Pivot the data - lines as orders and products as columns
df_pivot = pd.pivot_table(df1, index='order_id', columns='product_name', 
                    aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
#Transpose the pivot
df_pivot =df_pivot.T
#Create a column to count number of orders that the product appears
df_pivot['sum'] = df_pivot[df_pivot.columns].apply(lambda x: sum(x), axis = 1)
#Sort by number of orders. 
df_pivot.sort_values(by='sum', ascending=False, inplace=True)


# In[11]:


#Departments more frequent by itens sold
#Create a dataframe grouped by department
grouped = df_merge.groupby("department")["order_id"].agg(['count'])
#Rename columsn
grouped = grouped.rename({"count":"Ordersitens_bydept"}, axis=1)
#Reset index
grouped.reset_index(inplace = True)
#Create a column with the % of participation of each department in total itens sold
grouped['Ratio'] = grouped["Ordersitens_bydept"].apply(lambda x: x /grouped['Ordersitens_bydept'].sum())
#Sort by the departmens with more itens sold
grouped.sort_values(by='Ordersitens_bydept', ascending=False, inplace=True)
#Group some departments with low participation in one label callled others
grouped['department'] = grouped['department'].replace(['personal care', 'babies', 'international','alcohol','pets', 
                                                       'missing','other','bulk'],
                                                      'others')


# In[12]:


# Create dataframe to plot the number of orders by departments
df2 = df_merge.drop(["product_id", "department_id", 'add_to_cart_order','reordered','product_name'], axis=1)
# Pivot the data - lines as orders and products as columns
df_pivot_dept = pd.pivot_table(df2, index='order_id', columns='department', 
                    aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
#Transpose the pivot
df_pivot_dept =df_pivot_dept.T
#Create a column to count number of orders that the department appears
df_pivot_dept['sum'] = df_pivot_dept[df_pivot_dept.columns].apply(lambda x: sum(x), axis = 1)
#Sort by number of orders. 
df_pivot_dept.sort_values(by='sum', ascending=False, inplace=True)


# In[13]:


#Create a dataframe groupped by reordered or not
prod_reord = df_merge.groupby("product_name")["reordered"].agg(['count','sum'])
#Rename the columns
prod_reord = prod_reord.rename({"count":"ordered_sum"}, axis=1)
prod_reord = prod_reord.rename({"sum":"reordered_sum"}, axis=1)
#Reset index
prod_reord.reset_index(inplace = True)
#Create a new column to calculate the reordered probabibility (% of reordered)
prod_reord['reorder_probability'] = (prod_reord["reordered_sum"]/prod_reord['ordered_sum'])*100
#Sort values to show the most reordered
prod_reord.sort_values(by='reorder_probability', ascending=False, inplace=True)
#filter only the 30 most reordered
prod_reord_top30 = prod_reord[0:30]


# In[14]:


#Selected products
#The products that appears in at least 10% of orders. Represent more than 50% of total itens sold. 
list1 = prod_freq.index[0:30]
list2 = df_pivot.index[0:16]
products =  []
for variable in list1:
    if variable in list2:
        products.append(variable)


# In[15]:


#Create a dataframe to have products and dow, hour_of_day in the same dataframe
df_merge2 = pd.merge(df_merge, orders, how='left', on='order_id')


# In[16]:


#Create a function to return the number of orders by day of week or per hour of the day by product
def frequent_time_prod(product, option):
    df = df_merge2[df_merge2['product_name']==product]
    if option == "Day of week":
        freq = df['order_dow'].value_counts()
    elif option == 'Hour of the day':
        freq = df['order_hour_of_day'].value_counts()
    return freq


# ### Heatmap for the complementary products




# In[18]:


# Pivot the data - lines as orders and products as columns
pt = pd.pivot_table(df1, index='order_id', columns='product_name',
                    aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)


# In[19]:


# Create new support dataframe for two products only
#frequent_itemsets_two = apriori(pt, min_support=0.03, use_colnames=True,  max_len = 2)


# In[20]:


# Compute association rules for frequent_itemsets
#rulesConfidence_two = association_rules(frequent_itemsets_two, metric="confidence", min_threshold=0.50)

# High Confidence and high Lift - complementary products
#complementary_p = rulesConfidence_two[(rulesConfidence_two['lift'] > 1.3)].sort_values('lift', ascending=False)


# In[21]:


# Create column with the number of the left hand side items
#complementary_p['lhs items'] = complementary_p['antecedents'].apply(lambda x:len(x) )

# Create column with the number of the right hand side items
#complementary_p['rhs items'] = complementary_p['consequents'].apply(lambda x:len(x) )

# Replace frozen sets with strings
#complementary_p['antecedents_'] = complementary_p['antecedents'].apply(lambda a: ','.join(list(a)))
#complementary_p['consequents_'] = complementary_p['consequents'].apply(lambda a: ','.join(list(a)))

# Transform the DataFrame of rules into a matrix using the lift metric
#pivot = complementary_p.pivot(index = 'consequents_', columns = 'antecedents_', values= 'lift')


# In[22]:


# divide the data
#consequents = pivot.index.tolist()
#antecedents = pivot.columns.tolist()
#matrix = pivot.to_numpy()
#Define the data to plot
#data=go.Heatmap( x = antecedents, y = consequents, z = matrix, colorscale='YlOrRd')
#Define layout
#layout = dict(title=dict(
#                        text="Complementary Products - Lift"
#                        ),
#                  yaxis=dict(title="Consequents"),
#                  xaxis=dict(title="Antecedents"),
#                  title_x=0.5)
# create figure
#fig_heatmap = go.Figure( data = data, layout = layout )


# ### Creating the components

# In[23]:


#Create a  Plotly Visualization to show the days since prior order - Bar plot
#Define the dataset to be used
df = prior_orders
#Define the data to plot
data_bar = (dict(type='bar',
                     x=df.index,
                     y=df.values,
                     marker_color='lightseagreen',
                     showlegend=False))  
#Define the layout
layout_bar = dict(title=dict(
                        text='Days since prior order'),
                  xaxis=dict(title='Number of days'),
                  yaxis=dict(title='Number of orders'),
                  title_x=0.5)
fig_bar = go.Figure(data=data_bar, layout=layout_bar)


# In[24]:


##Create a Plotly Visualization to be included in dashboard - Bar plot
#Define the dataset to be used
df = number_orders
#Define the data to plot
data = dict(type='scatter',
                  x=df['number_orders'],
                  y=df['count'],
                  name='Quantity of users by quantity of orders'
                  )
#Define the layout
layout = dict(title=dict(text='Quantity of orders'),
                  xaxis=dict(title='Number of orders'),
                  yaxis=dict(title='Quantity of users')
                  )
#Create the figure
fig_users = go.Figure(data, layout)


# In[25]:


#Create a bar plot to show the basket size
#define the data
data = dict(type='bar',
                  x=avg_basket_top25['basket_size'],
                  y=avg_basket_top25['number_of_orders'],
                  name='Basket Size'
                  )
#define the layout
layout = dict(title=dict(text='Average Basket size'),
                  xaxis=dict(title='Basket Size'),
                  yaxis=dict(title='Number of orders')
                  )
#create the fig
avgbasket_fig = go.Figure(data, layout)


# In[26]:


#Products plots
#Plot number of itens sold by product - Treemap
#Define the dataset
df = prod_freq
#Define the data
treemap_trace = go.Treemap(labels=df.index[0:30], parents=[""] * len(df.index), values=df.values[0:30])
#Define the layout
treemap_layout = go.Layout( {"margin": dict(t=35, b=10, l=5, r=5, pad=4)},title=dict(text='Itens sold by product') )
#Create the figure
treemap_figure = go.Figure(data=treemap_trace, layout=treemap_layout)


# In[27]:


#Plotly number of orders by product - Treemap
#Define the dataset
df = df_pivot
#Define the data
treemap_trace = go.Treemap(labels=df.index[0:16], parents=[""] * len(df.index), values=df['sum'][0:16])
#Define the layout
treemap_layout = go.Layout({"margin": dict(t=35, b=10, l=5, r=5, pad=4)}, title=dict(text='Number of orders by product') )
#Create the figure
treemap_prod = go.Figure(data=treemap_trace, layout=treemap_layout)


# In[28]:


#Department plots
#Plotly % of itens sold by department - Pie plot
#Define the dataset
df = grouped
#Define the labels and the data
department_labels = df['department']
department_values = df['Ratio']
department_data = dict(type='pie',labels=department_labels,
                        values=department_values,)
#Define the layout
department_layout = dict(title=dict(text='Itens sold by department - % share'))
#Create the figure
department_fig = go.Figure(data= department_data, layout=department_layout)


# In[29]:


#Filter the  deparments that appears at least in 5% of orders
#Plotly number of orders by deparment
#Define the dataset
df = df_pivot_dept
#Define the data
treemap_trace = go.Treemap(labels=df.index[0:16], parents=[""] * len(df.index), values=df['sum'][0:16])
#Define the layout
treemap_layout = go.Layout({"margin": dict(t=30, b=10, l=5, r=5, pad=4)}, title=dict(text='Number of orders by department') )
#Create the figure
treemap_dept = go.Figure(data=treemap_trace, layout=treemap_layout)


# In[30]:


#Plotly Visualization - Bar plot
#Define the data
data_bar = (dict(type='bar',
                     y = prod_reord_top30['reorder_probability'],
                     x = prod_reord_top30['product_name'],
                     marker_color='lightseagreen',
                     showlegend=False
                    )
               )
#Define the layout 
layout_bar = dict(title=dict(
                        text='Products most reordered'
                        ),
                  yaxis=dict(title='% Reorder Probability'),
                  title_x=0.5,
                  
                 )
#Create the figure
reordered_bar = go.Figure(data=data_bar, layout=layout_bar)


# ### Designing the dashboard

# #### Layout

# In[31]:


app = dash.Dash(__name__,external_stylesheets =[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([
        dbc.Navbar(
            dbc.Row([dbc.Col(dbc.NavbarBrand('INSTACART MARKET BASKET ANALYSIS', className="ml-2"),
                style={"text-align":"center"}),],),
            color="LightBlue",
            dark=True,
            sticky="top",
        ),
             html.Br()
    ]),
    dcc.Tabs(id='tabs', value='tabStocks', persistence=False, children=[ 
        dcc.Tab(label='Consumer Behaviors', value='tabStocks', children=
            html.Div([
                html.Br([]),
                html.Div([
                    html.Div([
                    dbc.CardHeader(html.H4('Number of orders by Period')),
                    dbc.CardBody(
                        children =[
                        html.Div([html.Label('Choose an option:',style=dict(margin="10px")),
                        dcc.RadioItems(id='option',
                                       options=[{'label': 'Day of week', 'value': 'Day of week'},
                                                {'label': 'Hour of the day', 'value': 'Hour of the day'},],
                                       value='Day of week', labelStyle={'display': 'inline-block',})
                                 ], style=dict(display='flex', width = "100%")),
                        dcc.Graph(id='bar_plot', style={"height": "50%", "width": "100%"})])
                    ],style={"width": "100%"} ),
                    html.Div(id='margin', style={"width": "5%"}),
                    html.Div([
                        dbc.CardHeader(html.H4('Days since prior order')),
                        html.Br([]),
                        html.Br([]),
                        html.Div(dcc.Graph(id='prior_orders_plot', figure=fig_bar), style={"height": "50%", "width": "95%"}),],
                        style={"width": "100%"})
                ],style=dict(display='flex')),
                
                html.Br([]),
                html.Div([
                     html.Div([
                          dbc.CardHeader(html.H4('Basket Size')),
                          dcc.Graph(id='basket_plot',figure=avgbasket_fig, style={ "width": "100%"}),
                          ],style={"width": "100%"}),
                     html.Div(id='margin2', style={"width": "5%"}),
                     html.Div([
                          dbc.CardHeader(html.H4('Number of orders by users')),
                          html.Div(dcc.Graph(id='num_orders',figure=fig_users), style={"width": "95%"}),
                          ],style={"width": "100%"} )     
                    ], style=dict(display="flex"))
            ])),
            dcc.Tab(label='Products and Departmens Analysis', value='tabproducts',children=[
            html.Br([]),
            dbc.CardHeader(html.H4('Products analysis')),
            html.Br([]),
            html.Div([
                html.Div(dcc.Tabs(id='prods', value='prods_distr', persistence=False, children=[ 
                    dcc.Tab(label='Itens sold by product', value='prods_distr',children=
                        dcc.Graph(id='treemap_itens',figure=treemap_figure)),
                    dcc.Tab(label='Number of orders by product', value='tabproducts',children=
                        dcc.Graph(id='treemap_orders', figure=treemap_prod))]),style={"width":"100%"}), 
                html.Div(style={"width":"5%"}),
                html.Div(dcc.Graph(id='reordered_prod', figure=reordered_bar),style={"width":"100%"})
                    ],style=dict(display="flex",width="100%")),
            html.Br([]),
            html.Hr([]),
            html.Div([    
                html.Div([
                    
                        html.Label('Choose a product to analyze:',style=dict(margin="10px") ),
                        dcc.Dropdown(options=[{'label': k.capitalize(), 'value': k} for k in products],
                         placeholder="Select product",id='dropdownProducts', style=dict(width="40%",verticalAlign="middle")),
                        html.Label('Choose an option:',style=dict(margin="10px")),
                        dcc.RadioItems(id='option2',
                                       options=[{'label': 'Day of week', 'value': 'Day of week'},
                                                {'label': 'Hour of the day', 'value': 'Hour of the day'},],
                                       value='Day of week', labelStyle={'display': 'inline-block'})
                                 ], style=dict(display='flex', width = "100%")),
                dcc.Graph(id='bar_plot_prod', style={"height": "50%", "width": "100%"})]),
            html.Hr([]),
            #html.H4('Association Rules - Lift of complementary products'),
            html.Br([]),
            #dcc.Graph(figure=fig_heatmap, style={"width": "100%"}),
            dbc.CardHeader(html.H4('Departments participation')),
            html.Br([]),
            html.Div([
                      
                      html.Div(dcc.Graph(id='pie_graphic', figure=department_fig ),style={"width":"100%"}),
                      html.Div(style={"width":"5%"}),
                      html.Div(dcc.Graph(id='treemap_dept', figure=treemap_dept),style={"width":"100%"})       
                    ],style=dict(display="flex",width="100%")),]),
        
]),
                   
                    
    ],style=dict(margin='1em'))


# ### Callbacks

# In[32]:


#Plot Bar Transactions by period
@app.callback(Output('bar_plot', 'figure'),
            [Input('option', 'value')])
def update_barplot(option):
    data_bar = (dict(type='bar',
                     x=frequent_time(option).index,
                     y=frequent_time(option).values,
                     marker_color=' cornflowerblue',
                     showlegend=False))  
    layout_bar = dict(title=dict(
                        text=f'Frequency by {option}'),
                  xaxis=dict(title=f'{option}'),
                  yaxis=dict(title='Number of orders'),
                  title_x=0.5)
    fig_bar = go.Figure(data=data_bar, layout=layout_bar)
    return fig_bar


# In[33]:


#Plot Bar Transactions by period by product
@app.callback(Output('bar_plot_prod', 'figure'),
            [Input('dropdownProducts', 'value'),
             Input('option2', 'value')])
#Create a function with a bar plotly vizualization with the option by user choose 'day of week' or 'hour of the day'
#Also chose the product
def update_barplot_prod(product, option):
    data_bar = (dict(type='bar',
                     x=frequent_time_prod(product, option).index,
                     y=frequent_time_prod(product, option).values,
                     marker_color=' cornflowerblue',
                     showlegend=False))  
    layout_bar = dict(title=dict(
                        text=f'Frequency by {option} - {product}'),
                  xaxis=dict(title=f'{option}'),
                  yaxis=dict(title='Itens sold'),
                  title_x=0.5)
    fig_bar_prod = go.Figure(data=data_bar, layout=layout_bar)
    return fig_bar_prod


# In[34]:


if __name__ == '__main__':
    app.run_server()

