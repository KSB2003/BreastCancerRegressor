"""

Author: Kesav Bobba
Title: Breast cancer GUI application
Date: 16th september 2020

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tkinter import Tk, Button, Radiobutton, StringVar, Label, messagebox, Entry, DoubleVar

class DataPre():
    def __init__(self, filename):
        self.filename = filename
    def datareader(self):
        """

        :param filename: This is the data set that we would be analysing and going through.
        One thing to consider about the filename is that we have not taken variables that would have no effect such as serial number.
        :return: The code wold compile the y data in the form of a vector and the x data in the form of a matrix.
        """
        data = pd.read_csv('csIA.csv')
        xcolumn1 = ["radius_mean"]

        # setting the column names so that no mistakes are made in the data compilation and setting.
        self.xcolumn = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                   "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
                   "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                   "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                   "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                   "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
        self.ycolumn = ["diagnosis"]

        # in the next two lines we would be setting the x and the y data
        xdata = pd.read_csv(self.filename, usecols=self.xcolumn)
        ydata = pd.read_csv(self.filename, usecols=self.ycolumn)

        # finally over here we would be setting the x and the y data to proper variables and returning them in a matrice and a vector hence the '.values' at the end.
        self.x = xdata.iloc[:, :].values
        self.y = ydata.iloc[:, 0].values





    def splitterandmodify(self):
        """

        :param x: The x data is the independent variable from the data set that was created in the previous function.
        :param y: The y data is whether or nor the cancer was malignant or benign. This would be given to us in the form of a vector.
        :return: the train and test data with 'B' and 'M' being modified to 0 and 1.
        """

        """
        This step allows me to use logistic regression by converting the B's and the M's into 0s and 1s
        """
        for i in range(0, len(self.y)):
            if self.y[i] == 'B':
                self.y[i] = '0'
            else:
                self.y[i] = '1'

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, test_size=0.33, random_state=10)

        return self.xtrain, self.xtest, self.ytrain, self.ytest

    def scaler(self):
        """

        :param xtrain: the training dependent variables that have been modified accordingly.
        :return: the scaled values and the xtrain.
        """
        self.scale = StandardScaler()
        self.xtrain = self.scale.fit_transform(self.xtrain)
        return self.scale, self.xtrain



class logisticclassifier():
    def __init__(self):
        self.xcol = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                   "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
                   "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                   "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                   "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                   "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
    def classifier(self, xtrain, ytrain):
        """

        :param xtrain: the training dependent variables that have been modified accordingly.
        :param ytrain:the y train model that the system would be analysing.
        :return:
        """
        self.classer = LogisticRegression(random_state=10)

        self.classer.fit(xtrain, ytrain)

    def predictor(self, xtest):
        """
        :param xtest: the testing dependent variable.
        :param classer: It classifies and gives us a final reuslt based on the xtest.
        :return: it returns a vector of all of the ypredicted values.
        """

        ypredicted = self.classer.predict(xtest)
        return ypredicted


    def getweightage(self):
        """
        This class is determining the variables with the highest weightage in the result. By doing this we are able to select the variables for the short response for the best possible regressor.
        """
        weightages = self.classer.coef_[0]
        weightages = list(weightages)
        for i in range(0, len(weightages)):
            weightages[i] = abs(weightages[i])

        weightcopy = []

        for weight in weightages:
            weightcopy.append(weight)

        weightages.sort(reverse=True)

        biggest = weightages[0:5]

        varlist = []
        for m in range(0, len(biggest)):
            index = weightcopy.index(biggest[m])
            varlist.append(self.xcol[index])



        return varlist








def runner():

    """
    test runner, doesnt, return anything, just to check accuracy score and confusion matrix.
    :return:
    """
    refData = DataPre('csIA.csv')
    refData.datareader()
    xtrain, xtest, ytrain, ytest = refData.splitterandmodify()
    scale, xtrainref = refData.scaler()
    regress = logisticclassifier()
    regress.classifier(xtrainref, ytrain)
    ypredicted = regress.predictor(scale.transform(xtest))
    print(regress.getweightage())
    print(accuracy_score(ytest, ypredicted))
    print(confusion_matrix(ytest, ypredicted))





class GUI():
    """
    It is the GUI object, it cvontains all of the app formatting and is linked to the regressor in the above function.
    """
    def __init__(self, varlist, classer, scaler1):
        """

        This is the init function and it also intializes the variables that need to be looked at.
        """
        self.varlist = varlist
        # self.varlist1 = varlist1

        """
        This is the variable list, it contains all of the variables for the long answer, it contains a total of 30 variables all of which have some degree of weightage and 
        play roles in the determination of whether or not someone has breast cancer. 
        """
        self.varlist1 = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                    "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
                    "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                    "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
        self.classer = classer
        self.scaler1 = scaler1




        """
        This is the first GUI object, it is the opening page that offers the option between the long answer response and the short answer response. 
        """
        guiObj = Tk()
        guiObj.title('BREAST CANCER PREDICTOR')
        guiObj.configure(background='#ff00c3')
        guiObj.geometry('600x300')
        guiObj.resizable(width = False, height = False)


        """FIRST PAGE"""
        shortbuttonp1 = Button(guiObj, text="Click here for the short option", height = 5, fg='#1e00ff', command=self.shortsurvey)
        shortbuttonp1.grid(row=0, column=0, padx=20, pady=100)
        "------------------------------------------"
        Longbuttonp1 = Button(guiObj, text="Click here for a long option", height=5, fg='#1e00ff', command = self.longsurvey)
        Longbuttonp1.grid(row=0, column=3, padx=125, pady=15)



        guiObj.mainloop()


    def shortsurvey(self):

        """
        This method contains the page for the short function, every GUI object on Tkinter is a new page, and button require commands to be able to work.
        That is why short survey is the command to the short survey button.
        """
        guiObj2 = Tk()
        guiObj2.title("Short survey for breast cancer prediction")
        guiObj2.configure(background='#ff00c3')
        guiObj2.geometry('1200x600')
        guiObj2.resizable(width=False, height=False)

        label1 = Label(guiObj2, text=self.varlist[0])
        label2 = Label(guiObj2, text=self.varlist[1])
        label3 = Label(guiObj2, text=self.varlist[2])
        label4 = Label(guiObj2, text=self.varlist[3])
        label5 = Label(guiObj2, text=self.varlist[4])

        label1.grid(row=0, column=0, pady=30)
        label2.grid(row=1, column=0, pady=30)
        label3.grid(row=2, column=0, pady=30)
        label4.grid(row=3, column=0, pady=30)
        label5.grid(row=4, column=0, pady=30)


        self.doubleshort1 = DoubleVar()
        self.doubleshort2 = DoubleVar()
        self.doubleshort3 = DoubleVar()
        self.doubleshort4 = DoubleVar()
        self.doubleshort5 = DoubleVar()


        entry1 = Entry(guiObj2, textvariable=self.doubleshort1)
        entry2 = Entry(guiObj2, textvariable=self.doubleshort2)
        entry3 = Entry(guiObj2, textvariable=self.doubleshort3)
        entry4 = Entry(guiObj2, textvariable=self.doubleshort4)
        entry5 = Entry(guiObj2, textvariable=self.doubleshort5)


        entry1.grid(row=0, column=1, padx=10)
        entry2.grid(row=1, column=1, padx=10)
        entry3.grid(row=2, column=1, padx=10)
        entry4.grid(row=3, column=1, padx=10)
        entry5.grid(row=4, column=1, padx=10)

        submit = Button(guiObj2, text='SUBMIT', fg='#1e00ff', command=self.regressnewpage)
        submit.grid(row=6, column=3, padx=260)



    def regressnewpage(self):
        regressObj = LogisticRegression(random_state=10)

        xcolumn = self.varlist
        xdata = pd.read_csv('csIA.csv', usecols=xcolumn)


        ycolumn = ['diagnosis']
        ydata = pd.read_csv('csIA.csv', usecols=ycolumn)

        x = xdata.iloc[:, :].values
        y = ydata.iloc[:, 0].values

        for i in range(0, len(y)):
            if y[i]=='B':
                y[i] = '0'
            else:
                y[i] = '1'


        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=10)

        scaler = StandardScaler()
        xtrainmod = scaler.fit_transform(xtrain)

        regressObj.fit(xtrainmod, ytrain)

        listinput = [[float(self.doubleshort1.get()), float(self.doubleshort2.get()), float(self.doubleshort3.get()), float(self.doubleshort4.get()), float(self.doubleshort5.get())]]




        ypredicted = regressObj.predict(xtest)


        result = regressObj.predict(scaler.transform(listinput))
        print(result)




        if result==0:
            result = "YOU ARE SAFE! MAKE SURE TO STAY HEALTHY!"
        else:
            result = "You are malignant. Please consult a doctor immediately."




        messagebox.showinfo("RESULTS", result)





    def longsurvey(self):

        """
        This is the page for the long regressor, we must use this pager in order to be able to enter in the long values
        with the help of the long values which may not always be available we can get the most accurate possible result,
        in fact the accuracy may range upto 97.3% which is better just to make sure.
        """
        guiObj3 = Tk()
        guiObj3.title("Long data response for breast cancer prediction.")
        guiObj3.configure(background='#ff00c3')
        guiObj3.geometry('1200x600')
        guiObj3.resizable(width=False, height=False)

        label1 = Label(guiObj3, text=self.varlist1[0])
        label1.grid(row=0, column=0)

        label2 = Label(guiObj3, text=self.varlist1[1])
        label2.grid(row=1, column=0)

        label3 = Label(guiObj3, text=self.varlist1[2])
        label3.grid(row=2, column=0)

        label4 = Label(guiObj3, text=self.varlist1[3])
        label4.grid(row=3, column=0)

        label5 = Label(guiObj3, text=self.varlist1[4])
        label5.grid(row=4, column=0)

        label6 = Label(guiObj3, text=self.varlist1[5])
        label6.grid(row=5, column=0)

        label7 = Label(guiObj3, text=self.varlist1[6])
        label7.grid(row=6, column=0)

        label8 = Label(guiObj3, text=self.varlist1[7])
        label8.grid(row=7, column=0)

        label9 = Label(guiObj3, text=self.varlist1[8])
        label9.grid(row=8, column=0)

        label10 = Label(guiObj3, text=self.varlist1[9])
        label10.grid(row=9, column=0)

        label11 = Label(guiObj3, text=self.varlist1[10])
        label11.grid(row=10, column=0)

        label12 = Label(guiObj3, text=self.varlist1[11])
        label12.grid(row=11, column=0)

        label13 = Label(guiObj3, text=self.varlist1[12])
        label13.grid(row=0, column=2)

        label14 = Label(guiObj3, text=self.varlist1[13])
        label14.grid(row=1, column=2)

        label15 = Label(guiObj3, text=self.varlist1[14])
        label15.grid(row=2, column=2)

        label16 = Label(guiObj3, text=self.varlist1[15])
        label16.grid(row=3, column=2)

        label17 = Label(guiObj3, text=self.varlist1[16])
        label17.grid(row=4, column=2)

        label18 = Label(guiObj3, text=self.varlist1[17])
        label18.grid(row=5, column=2)

        label19 = Label(guiObj3, text=self.varlist1[18])
        label19.grid(row=6, column=2)

        label20 = Label(guiObj3, text=self.varlist1[19])
        label20.grid(row=7, column=2)

        label21 = Label(guiObj3, text=self.varlist1[20])
        label21.grid(row=8, column=2)

        label22 = Label(guiObj3, text=self.varlist1[21])
        label22.grid(row=9, column=2)

        label23 = Label(guiObj3, text=self.varlist1[22])
        label23.grid(row=10, column=2)

        label24 = Label(guiObj3, text=self.varlist1[23])
        label24.grid(row=11, column=2)

        label25 = Label(guiObj3, text=self.varlist1[24])
        label25.grid(row=0, column=4)

        label26 = Label(guiObj3, text=self.varlist1[25])
        label26.grid(row=1, column=4)

        label27 = Label(guiObj3, text=self.varlist1[26])
        label27.grid(row=2, column=4)

        label28 = Label(guiObj3, text=self.varlist1[27])
        label28.grid(row=3, column=4)

        label29 = Label(guiObj3, text=self.varlist1[28])
        label29.grid(row=4, column=4)

        label30 = Label(guiObj3, text=self.varlist1[29])
        label30.grid(row=5, column=4)

        self.double1 = DoubleVar()
        self.double2 = DoubleVar()
        self.double3 = DoubleVar()
        self.double4 = DoubleVar()
        self.double5 = DoubleVar()
        self.double6 = DoubleVar()
        self.double7 = DoubleVar()
        self.double8 = DoubleVar()
        self.double9 = DoubleVar()
        self.double10 = DoubleVar()
        self.double11 = DoubleVar()
        self.double12 = DoubleVar()
        self.double13 = DoubleVar()
        self.double14 = DoubleVar()
        self.double15 = DoubleVar()
        self.double16 = DoubleVar()
        self.double17 = DoubleVar()
        self.double18 = DoubleVar()
        self.double19 = DoubleVar()
        self.double20 = DoubleVar()
        self.double21 = DoubleVar()
        self.double22 = DoubleVar()
        self.double23 = DoubleVar()
        self.double24 = DoubleVar()
        self.double25 = DoubleVar()
        self.double26 = DoubleVar()
        self.double27 = DoubleVar()
        self.double28 = DoubleVar()
        self.double29 = DoubleVar()
        self.double30 = DoubleVar()

        entry1 = Entry(guiObj3, textvariable=self.double1)
        entry1.grid(row=0, column=1)

        entry2 = Entry(guiObj3, textvariable=self.double2)
        entry2.grid(row=1, column=1)

        entry3 = Entry(guiObj3, textvariable=self.double3)
        entry3.grid(row=2, column=1)

        entry4 = Entry(guiObj3, textvariable=self.double4)
        entry4.grid(row=3, column=1)

        entry5 = Entry(guiObj3, textvariable=self.double5)
        entry5.grid(row=4, column=1)

        entry6 = Entry(guiObj3, textvariable=self.double6)
        entry6.grid(row=5, column=1)

        entry7 = Entry(guiObj3, textvariable=self.double7)
        entry7.grid(row=6, column=1)

        entry8 = Entry(guiObj3, textvariable=self.double8)
        entry8.grid(row=7, column=1)

        entry9 = Entry(guiObj3, textvariable=self.double9)
        entry9.grid(row=8, column=1)

        entry10 = Entry(guiObj3, textvariable=self.double10)
        entry10.grid(row=9, column=1)

        entry11 = Entry(guiObj3, textvariable=self.double11)
        entry11.grid(row=10, column=1)

        entry12 = Entry(guiObj3, textvariable=self.double12)
        entry12.grid(row=11, column=1)

        entry13 = Entry(guiObj3, textvariable=self.double13)
        entry13.grid(row=0, column=3)

        entry14 = Entry(guiObj3, textvariable=self.double14)
        entry14.grid(row=1, column=3)

        entry15 = Entry(guiObj3, textvariable=self.double15)
        entry15.grid(row=2, column=3)

        entry16 = Entry(guiObj3, textvariable=self.double16)
        entry16.grid(row=3, column=3)

        entry17 = Entry(guiObj3, textvariable=self.double17)
        entry17.grid(row=4, column=3)

        entry18 = Entry(guiObj3, textvariable=self.double18)
        entry18.grid(row=5, column=3)

        entry19 = Entry(guiObj3, textvariable=self.double19)
        entry19.grid(row=6, column=3)

        entry20 = Entry(guiObj3, textvariable=self.double20)
        entry20.grid(row=7, column=3)

        entry21 = Entry(guiObj3, textvariable=self.double21)
        entry21.grid(row=8, column=3)

        entry22 = Entry(guiObj3, textvariable=self.double22)
        entry22.grid(row=9, column=3)

        entry23 = Entry(guiObj3, textvariable=self.double23)
        entry23.grid(row=10, column=3)

        entry24 = Entry(guiObj3, textvariable=self.double24)
        entry24.grid(row=11, column=3)

        entry25 = Entry(guiObj3, textvariable=self.double25)
        entry25.grid(row=0, column=5)

        entry26 = Entry(guiObj3, textvariable=self.double26)
        entry26.grid(row=1, column=5)

        entry27 = Entry(guiObj3, textvariable=self.double27)
        entry27.grid(row=2, column=5)

        entry28 = Entry(guiObj3, textvariable=self.double28)
        entry28.grid(row=3, column=5)

        entry29 = Entry(guiObj3, textvariable=self.double29)
        entry29.grid(row=4, column=5)

        entry30 = Entry(guiObj3, textvariable=self.double30)
        entry30.grid(row=5, column=5)

        submitbutton = Button(guiObj3, text='SUBMIT', fg='#1e00ff', command=self.regressnewpage_long)
        submitbutton.grid(column=3, row=13, pady=200)



        





    def regressnewpage_long(self):
        independentvarlist = [float(self.double1.get()), float(self.double2.get()), float(self.double3.get()), float(self.double4.get()), float(self.double5.get()),
                             float(self.double6.get()), float(self.double7.get()), float(self.double8.get()), float(self.double9.get()), float(self.double10.get()),
                             float(self.double11.get()), float(self.double12.get()), float(self.double13.get()), float(self.double14.get()), float(self.double15.get()),
                             float(self.double16.get()), float(self.double17.get()), float(self.double18.get()), float(self.double19.get()), float(self.double20.get()),
                             float(self.double21.get()), float(self.double22.get()), float(self.double23.get()), float(self.double24.get()), float(self.double25.get()),
                             float(self.double26.get()), float(self.double27.get()), float(self.double28.get()), float(self.double29.get()), float(self.double30.get())]
        independentvarlist = self.scaler1.transform([independentvarlist])
        ypredicted = self.classer.predict(independentvarlist)

        result = None
        if ypredicted==0:
            result = "YOU ARE SAFE! MAKE SURE TO STAY HEALTHY!"
        else:
            result = "You are malignant. Please consult a doctor immediately."

        messagebox.showinfo("RESULTS", result)







def runner1():
    """
    This part of the code is called the runner, it runs the code for the regressor and gives the coder control over the
    program so that he can run it whenever he wants to.
    """
    refData = DataPre('csIA.csv')
    refData.datareader()
    xtrain, xtest, ytrain, ytest = refData.splitterandmodify()
    scale, xtrainref = refData.scaler()
    regress = logisticclassifier()
    regress.classifier(xtrainref, ytrain)
    ypredicted = regress.predictor(scale.transform(xtest))
    # print(regress.getweightage())
    # print(accuracy_score(ytest, ypredicted))
    # print(confusion_matrix(ytest, ypredicted))
    GUI(['radius_se', 'texture_worst', 'compactness_se', 'radius_worst', 'concavity_worst'], regress.classer, scale)

runner1()

#17.99	10.38	122.8	1001	0.1184	0.2776	0.3001	0.1471	0.2419	0.0787	1.095	0.9053	8.589	153.4	0.0064	0.049	0.0537	0.0159	0.03	0.0062	25.38	17.33	184.6	2019	0.1622	0.6656	0.7119	0.2654	0.4601	0.1189







#test set for short survey: 1.095, 17.33, 0.049, 25.38, 0.7119