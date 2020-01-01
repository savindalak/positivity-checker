import tkinter as tk
import requests
from bs4 import BeautifulSoup

window = tk.Tk()
window.title('Multi website positivity checker')
window.geometry('1000x600')

list_of_websites = list()
list_of_pos_neg = list()
dic_of_results = dict()


# function for yes button-----------------------
def append_list():
    if not str(entry1.get()) == '':
        list_of_websites.append(str(entry1.get()))
        entry1.delete(0, 'end')


# define positive negative ratio function---------------------
def pos_neg_ratio(url, positive_words, negative_words):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.text
    except:
        print('error in site url request')
        return

    pos = neg = 0

    for word in text.split(' '):

        if word in positive_words:
            pos += 1

        if word in negative_words:
            neg += 1
    if neg != 0:
        return (pos / neg)
    else:
        return 'No negative comments found'


# define function to get positive and negative words from the website
def get_words(url):
    try:
        words = requests.get(url).content.decode('latin-1')
        word_list = words.split('\n')
        # print('response received')
    except:
        print('error in getting words request')
        return
    index = 0
    while index < len(word_list):
        word = word_list[index]
        if ';' in word or not word:
            word_list.pop(index)
        else:
            index += 1
    return word_list


# get the positive and negative words and store
def pos_neg_word_lists():
    p_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
    n_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'
    positive_words_inapp = get_words(p_url)
    negative_words_inapp = get_words(n_url)
    if len(positive_words_inapp) == 0 or len(negative_words_inapp) == 0:
        return 'pos or neg words list is empty!'

    if positive_words_inapp == None or negative_words_inapp == None:
        return 'pos or neg words didnot load properly!'
    return positive_words_inapp, negative_words_inapp


# iterate over list of web addresses
def get_pos_neg_ratio():
    if not str(entry1.get()) == '':
        list_of_websites.append(str(entry1.get()))

    for site in list_of_websites:
        ratio = pos_neg_ratio(site, pos_neg_word_lists()[0], pos_neg_word_lists()[1])
        list_of_pos_neg.append(ratio)


def positivity(x):
    try:
        if x > 4:
            return 'Very positive'
        if x > 1:
            return 'positive'
        if x < 0.2:
            return 'Very negative'
        else:
            return 'Negative'
    except:
        print('some error occured')


def create_dictionary():
    dic_of_results['Name of website'] = list_of_websites
    dic_of_results['Positive/negative ratio'] = list_of_pos_neg
    ratings_list = []
    for i in range(len(list_of_pos_neg)):
        ratings_list.append(positivity(dic_of_results['Positive/negative ratio'][i]))

    dic_of_results['Rating'] = ratings_list
    return dic_of_results


def results_display():
    get_pos_neg_ratio()

    if len(list_of_websites) != 0 or len(list_of_pos_neg) != 0:
        dic_final = create_dictionary()
        # -----create text field-------------------
        rows = len(list_of_websites)
        columns = 3

        for i in range(rows):
            result_display_ratio = tk.Text(master=window, height=3, width=40, bg='#fff830')
            result_display_ratio.grid(row=i + 3, column=0)
            result_display_ratio.insert(tk.END, str(dic_final['Name of website'][i]))

            result_display_ratio = tk.Text(master=window, height=3, width=40, bg='#fff830')
            result_display_ratio.grid(row=i + 3, column=1)
            result_display_ratio.insert(tk.END, str(dic_final['Positive/negative ratio'][i]))

            result_display_ratio = tk.Text(master=window, height=3, width=40, bg='#fff830')
            result_display_ratio.grid(row=i + 3, column=2)
            result_display_ratio.insert(tk.END, str(dic_final['Rating'][i]))
        list_of_pos_neg.clear()
        list_of_websites.clear()

    # result_display_sites=tk.Text(master=window,height=10,width=30,bg='#ADD8E6')
    # result_display_sites.grid(column=0,row=3)
    # result_display_sites.insert(tk.END,list_of_websites)


# -------labels-----------
label1 = tk.Label(text='Enter URLs here ', font=('verdana', 10), width=40)
label1.grid(row=0, column=0)

label2 = tk.Label(text='Do you want to add more websites', font=('verdana', 10), width=40)
label2.grid(row=1, column=0)

label3 = tk.Label(text='your websites', font=('verdana', 10), width=40)
label3.grid(row=2, column=0)

label4 = tk.Label(text='positive/negative ratio', font=('verdana', 10), width=40)
label4.grid(row=2, column=1)

label5 = tk.Label(text='Ratings', font=('verdana', 10), width=40)
label5.grid(row=2, column=2)

# -------Entry feilds-------------
entry1 = tk.Entry(width=55)
entry1.grid(row=0, column=1)

# ------------buttons---------------
f1 = tk.Frame(window)

button1 = tk.Button(f1, text='YES', command=append_list)
button2 = tk.Button(f1, text='NO,Show me the results', command=results_display)
f1.grid(row=1, column=1)
button1.pack(side='left')
button2.pack(side='left')

window.configure(bg='#ADD8E6')
window.mainloop()
