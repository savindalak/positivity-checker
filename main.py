import tkinter as tk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import datetime
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

matplotlib.use('TkAgg')

csv_file = 'immi_data_joined_sql_sorted.csv'

window = tk.Tk()
window.geometry('1000x600')
window.maxsize(width=1000, height=600)
window.title('489 VISA GRANT DAY PREDICTOR')

# ---------images-----------------
image1 = ImageTk.PhotoImage(Image.open('1280px-Flag_of_Australia.svg.png'))
label_image = tk.Label(image=image1, bd=0)
label_image.place(x=0, y=0)


# my_data=pd.DataFrame({'lodgement_date':['Jun 11, 2019'],'occupation':[233512],'sponsoring_state':[4],'onshore_offshore':[0],'agent_or_not':[1]})

def my_data():
    my_data_dic = dict()
    my_data_dic['lodgement_date'] = [str(entry1.get())]
    my_data_dic['occupation'] = [float(entry2.get())]
    my_data_dic['sponsoring_state'] = [float(entry3.get())]
    my_data_dic['onshore_offshore'] = [float(entry4.get())]
    my_data_dic['agent_or_not'] = [float(entry5.get())]

    return my_data_dic


def get_visa_grant_dates(csv_file):
    my_visa_data = pd.DataFrame(my_data())
    big_data = pd.read_csv(csv_file)
    big_data['lodgement_date'] = big_data['lodgement_date'].astype('datetime64[ns]')
    big_data['visa_granted_date'] = big_data['visa_granted_date'].astype('datetime64[ns]')
    big_data['occupation'] = big_data['occupation'].astype(int, errors='ignore')
    selected_data = big_data[
        ['lodgement_date', 'visa_granted_date', 'occupation', 'sponsoring_state', 'visa_grant_in_days',
         'onshore_offshore', 'agent_or_not']]
    selected_by_visagrant_date = selected_data.dropna(subset=['visa_granted_date'])
    df_by_occupation = selected_by_visagrant_date.replace({'occupation': 'Other'}, {'occupation': np.nan}).dropna(
        subset=['occupation'])
    df_by_state = df_by_occupation.dropna(subset=['sponsoring_state'])
    df_by_state = df_by_state.dropna(subset=['visa_grant_in_days'])
    df_by_state['occupation'] = df_by_state['occupation'].astype(int)
    df_by_state['visa_grant_in_days'] = df_by_state['visa_grant_in_days'].astype(int)
    df_by_onshore_offshore = df_by_state.dropna(subset=['onshore_offshore'])
    df_by_agent = df_by_onshore_offshore.dropna(subset=['agent_or_not'])

    data_for_predictions = df_by_agent
    sc = StandardScaler()
    rfc = RandomForestClassifier(n_estimators=200)
    lm = LinearRegression()

    immi_variables = data_for_predictions[['occupation', 'sponsoring_state', 'onshore_offshore', 'agent_or_not']]
    visa_grant_dates = pd.to_numeric(data_for_predictions['visa_grant_in_days'])
    # immi_variables['lodgement_date']=pd.to_numeric(immi_variables['lodgement_date'])
    immi_variables['sponsoring_state'] = immi_variables['sponsoring_state'].astype('category').cat.codes
    immi_variables['onshore_offshore'] = immi_variables['onshore_offshore'].astype('category').cat.codes
    immi_variables['agent_or_not'] = immi_variables['agent_or_not'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(immi_variables, visa_grant_dates, test_size=0.3,
                                                        random_state=40)
    X_train_transformed = sc.fit_transform(X_train)
    X_test_transformed = sc.transform(X_test)
    rfc.fit(X_train_transformed, y_train)
    pearson_value_rfc = scipy.stats.pearsonr(rfc.predict(X_test_transformed), y_test)
    rfc_y_values = rfc.predict(X_test_transformed)

    clf = svm.SVC()
    clf.fit(X_train_transformed, y_train)
    clf.predict(X_test_transformed)
    pearson_value_clf = scipy.stats.pearsonr(clf.predict(X_test_transformed), y_test)
    clf_y_values = clf.predict(X_test_transformed)

    mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500)
    mlpc.fit(X_train_transformed, y_train)
    mlpc.predict(X_test_transformed)
    pearson_value_mlpc = scipy.stats.pearsonr(mlpc.predict(X_test_transformed), y_test)
    mlpc_y_values = mlpc.predict(X_test_transformed)

    my_visa_data['lodgement_date'] = my_visa_data['lodgement_date'].astype('datetime64[ns]')
    my_data_for_predictions = my_visa_data[['occupation', 'sponsoring_state', 'onshore_offshore', 'agent_or_not']]
    my_data_transformed = sc.transform(my_data_for_predictions)

    my_visa_predictions_clf = clf.predict(my_data_transformed)
    clf_predicted_date = pd.to_timedelta(str(my_visa_predictions_clf[0]) + ' days') + my_visa_data['lodgement_date']

    my_visa_predictions_rfc = rfc.predict(my_data_transformed)
    rfc_predicted_date = pd.to_timedelta(str(my_visa_predictions_rfc[0]) + ' days') + my_visa_data['lodgement_date']

    my_visa_predictions_mlpc = mlpc.predict(my_data_transformed)
    mlpc_predicted_date = pd.to_timedelta(str(my_visa_predictions_mlpc[0]) + ' days') + my_visa_data['lodgement_date']

    results = dict()
    results['pearson_value_rfc & predicted_date'] = [pearson_value_rfc,
                                                     rfc_predicted_date.dt.strftime('%Y-%m-%d').iat[0]]
    results['pearson_value_clf & & predicted_date'] = [pearson_value_clf,
                                                       clf_predicted_date.dt.strftime('%Y-%m-%d').iat[0]]
    results['pearson_value_mlpc & predicted_date'] = [pearson_value_mlpc,
                                                      mlpc_predicted_date.dt.strftime('%Y-%m-%d').iat[0]]
    results['data_summary'] = data_for_predictions.count()

    return results, rfc_y_values, clf_y_values, mlpc_y_values, y_test,


# ---------draw graph----------------------------
def draw_graphs(results):
    f = Figure(figsize=(4, 4), dpi=100)
    a = f.add_subplot(111)
    sns.distplot(results[4], ax=a, hist=False, label='test values')
    sns.distplot(results[1], ax=a, hist=False, label='RFC method')
    sns.distplot(results[2], ax=a, hist=False, label='CLF method')
    sns.distplot(results[3], ax=a, hist=False, label='MLPC method')

    a.set_xlabel('No of days')
    a.legend()
    a.set_title('Accuracy checking plots')

    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=6, column=2, rowspan=4)


def show_results():
    try:

        datetime.datetime.strptime(str(entry1.get()), '%b %d, %Y')

    except:
        text_feild1 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=2)
        text_feild1.grid(column=1, row=6)
        text_feild1.insert(tk.END, 'Enter valid date format')
        return
    if entry2.get() == '':
        text_feild1 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=2)
        text_feild1.grid(column=1, row=6)
        text_feild1.insert(tk.END, 'Enter valid occupation code')
        return
    if entry3.get() == '' or entry4.get() == '' or entry5.get() == '':
        text_feild1 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=2)
        text_feild1.grid(column=1, row=6)
        text_feild1.insert(tk.END, 'Enter valid numbers')
        return

    if float(entry3.get()) in [1, 2, 3, 4, 5, 6] and float(entry4.get()) in [0, 1] and float(entry5.get()) in [1, 0]:
        results = get_visa_grant_dates(csv_file)
    else:
        text_feild1 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=2)
        text_feild1.grid(column=1, row=6)
        text_feild1.insert(tk.END, 'Enter valid values')
        return
    # -------create text feilds-----------------
    text_feild1 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=4)
    text_feild1.grid(column=1, row=6)
    text_feild1.insert(tk.END, results[0]['pearson_value_rfc & predicted_date'][1])

    text_feild2 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=4)
    text_feild2.grid(column=1, row=7)
    text_feild2.insert(tk.END, results[0]['pearson_value_mlpc & predicted_date'][1])

    text_feild3 = tk.Text(master=window, font=10, width=20, bg='#fff830', height=4)
    text_feild3.grid(column=1, row=8)
    text_feild3.insert(tk.END, results[0]['pearson_value_clf & & predicted_date'][1])

    draw_graphs(results)


# ------Labels--------------------
label1 = tk.Label(text='Enter the date of lodgement here', font=('verdana', 12), width=30)
label1.grid(row=0, column=0)

label2 = tk.Label(text='ANZCO code of occupation', font=('verdana', 12), width=30)
label2.grid(row=1, column=0)

label3 = tk.Label(text='sponsoring state', font=('verdana', 12), width=30)
label3.grid(row=2, column=0)

label4 = tk.Label(text='TAS=4,SA=3,VIC=5,WA=6,NSW=0,QLD=2,NT=1', font=('verdana', 12), width=40)
label4.grid(row=2, column=2)

label5 = tk.Label(text='Onshore or offshore', font=('verdana', 12), width=30)
label5.grid(row=3, column=0)

label6 = tk.Label(text='Onshore=1 , Offshore=0', font=('verdana', 12), width=40)
label6.grid(row=3, column=2)

label7 = tk.Label(text='Agent or not', font=('verdana', 12), width=30)
label7.grid(row=4, column=0)

label8 = tk.Label(text='Agent=1 , Individual=0', font=('verdana', 12), width=40)
label8.grid(row=4, column=2)

label9 = tk.Label(text='RFC prediction', font=('verdana', 12), width=30, height=4)
label9.grid(row=6, column=0)

label10 = tk.Label(text='Enter in this format (Jun 11, 2019)', font=('verdana', 12), width=40)
label10.grid(row=0, column=2)

label11 = tk.Label(text='MLPC prediction', font=('verdana', 12), width=30, height=4)
label11.grid(row=7, column=0)

label12 = tk.Label(text='CLF prediction', font=('verdana', 12), width=30, height=4)
label12.grid(row=8, column=0)

label13 = tk.Label(
    text='Pick the closest matching curve\n to the blue curve(test values)\n from the 3 curves(RFC,MLPC & CLF)\n and select the most accurate\n predicted date from \nabove 3 values for your case',
    font=('verdana', 12), width=30, height=7, bg='#fff830')
label13.grid(row=9, column=0)

# -------Entry---------------
entry1 = tk.Entry(width=20, font=11)
entry1.grid(row=0, column=1)

entry2 = tk.Entry(width=20, font=11)
entry2.grid(row=1, column=1)

entry3 = tk.Entry(width=20, font=11)
entry3.grid(row=2, column=1)

entry4 = tk.Entry(width=20, font=11)
entry4.grid(row=3, column=1)

entry5 = tk.Entry(width=20, font=11)
entry5.grid(row=4, column=1)

# ------------buttons--------------
button1 = tk.Button(text='Show results', font=('verdana', 10), bg='#fff830', command=show_results)
button1.grid(row=5, column=1)

window.configure(bg='#00008B')
window.mainloop()
