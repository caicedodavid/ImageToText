## CRNN API

Creates an API of a CRNN model implementation from https://github.com/kurapan/CRNN

#### Build and run
```
docker build . -t crnn-api:latest 
```
```
docker run -it -p 4242:5000 crnn-api:latest 
```

### Api
The api uses GraphQL so you should open a browser and go to ``localhost:4242/graphql`` then add the following query:
```
query textInImage($url:String!) {
  textInImage(url:$url)
}
```
with the following query variables:
```
{"url": "https://your.image.url.jpg"}
```
