## clasify_images_api
<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
![GitHub](https://img.shields.io/npm/v/npm)
![GitHub repo size](https://img.shields.io/github/repo-size/lmt20/clasify_images_api)
![GitHub issues](https://img.shields.io/github/issues/lmt20/clasify_images_api)
![GitHub contributors](https://img.shields.io/github/contributors/lmt20/clasify_images_api)
![Twitter Follow](https://img.shields.io/twitter/follow/TruongLeManh?style=social)

INTRODUCTION
------------
- This API is created to categorize an image (currently i am setting to only work with animal images of 5 categories: butterfly, chicken, spider, elephant, parrot.
- I created this API for search features using images and suggestions features in my upcoming apps (create a small website version like pixabay.com app)
- I used Flask to create this API and use opencv, sklearn, pyspark library to implement some algorithms to create the classifier.

REQUIREMENTS
------------

INSTALLATION
------------

CONFIGURATION AND USAGE
-------------
1. test connection
- GET: https://classify-image-api.herokuapp.com/test 
2. using
- POST: https://classify-image-api.herokuapp.com/images 
- body: 
ex:
{
    "urlPath": "https://sohanews.sohacdn.com/thumb_w/660/2019/12/24/vet-15771805512831137885881-crop-15771805550711648518742.jpg"
}

AUTHOR
-----------
👤 **Le Manh Truong**
* Twitter: [@TruongLeManh](https://twitter.com/TruongLeManh)
* Github: [@lmt20](https://github.com/lmt20)
* LinkedIn: [@truong-le-manh](https://www.linkedin.com/in/truong-le-manh/)

