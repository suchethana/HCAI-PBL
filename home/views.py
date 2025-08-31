# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Suchethana Swaroopa Putta Narasaiah", "matriculation": "607087"},
        #{"name": "John Smith", "matriculation": "654321"},
        #{"name": "Alex Johnson", "matriculation": "789012"},
    ]
    
    projects = [
        {"name": "Project 1: Automated Machine Learning", "url_name": "project1:index"},
        {"name": "Project 2: Active Learning for Text Classification", "url_name": "project2:index"},
        {"name": "Project 3: Explainability", "url_name": "project3:index"},
        {"name": "Project 4: Influence of future predictions over active learning of users’ tastes for recommender systems", "url_name": "project4:index"},
        {"name": "Project 5:  Reinforcement Learning with Human Feedback", "url_name": "project5:index"},

    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))