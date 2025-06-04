from django.shortcuts import render, redirect, get_object_or_404


import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.

from Remote_User.models import ClientRegister_Model,placement_prediction_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":

            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            phoneno = request.POST.get('phoneno')
            country = request.POST.get('country')
            state = request.POST.get('state')
            city = request.POST.get('city')
            address = request.POST.get('address')
            gender = request.POST.get('gender')
            ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                                country=country, state=state, city=city, address=address, gender=gender)
            obj = "Registered Successfully"
            return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Placement_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            RID= request.POST.get('RID')
            Age= request.POST.get('Age')
            Gender= request.POST.get('Gender')
            Stream= request.POST.get('Stream')
            Internships= request.POST.get('Internships')
            Btech_CGPA= request.POST.get('Btech_CGPA')
            SSLC_Percentage= request.POST.get('SSLC_Percentage')
            PUC_Percentage= request.POST.get('PUC_Percentage')
            Hostel= request.POST.get('Hostel')
            HistoryOfBacklogs= request.POST.get('HistoryOfBacklogs')
            Salary= request.POST.get('Salary')

        dataset = pd.read_csv('Datasets.csv')

        def apply_results(label):
            if (label == 0):
                return 0  # Not Placed
            elif (label == 1):
                return 1  # Placed

        dataset['results'] = dataset['PlacedOrNot'].apply(apply_results)
        cv = CountVectorizer()
        x = dataset["RID"]
        y = dataset["results"]
        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
        print(x)
        print("Y")
        print(y)

        x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        RID1 = [RID]
        vector1 = cv.transform(RID1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Not Placed'
        elif prediction == 1:
            val = 'Placed'

        print(val)
        print(pred1)

        placement_prediction_type.objects.create(
        RID=RID,
        Age=Age,
        Gender=Gender,
        Stream=Stream,
        Internships=Internships,
        Btech_CGPA=Btech_CGPA,
        SSLC_Percentage=SSLC_Percentage,
        PUC_Percentage=PUC_Percentage,
        Hostel=Hostel,
        HistoryOfBacklogs=HistoryOfBacklogs,
        Salary=Salary,
        Prediction=val)

        return render(request, 'RUser/Predict_Placement_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Placement_Type.html')



