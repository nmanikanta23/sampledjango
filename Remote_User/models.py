from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=3000)
    gender= models.CharField(max_length=30)


class placement_prediction_type(models.Model):


        RID= models.CharField(max_length=300)
        Age= models.CharField(max_length=300)
        Gender= models.CharField(max_length=300)
        Stream= models.CharField(max_length=300)
        Internships= models.CharField(max_length=300)
        Btech_CGPA= models.CharField(max_length=300)
        SSLC_Percentage= models.CharField(max_length=300)
        PUC_Percentage= models.CharField(max_length=300)
        Hostel= models.CharField(max_length=300)
        HistoryOfBacklogs= models.CharField(max_length=300)
        Salary= models.CharField(max_length=300)
        Prediction= models.CharField(max_length=300)



class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



