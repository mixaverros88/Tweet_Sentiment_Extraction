import { Classification } from '../common/models/classification';
import { RandomTweet } from '../common/models/randomTweet';
import { Component, Injectable  } from '@angular/core';
import { HttpClient} from '@angular/common/http';
import {ConstantsService} from '../common/services/constants.service';

@Component({
  selector: 'app-classification',
  templateUrl: './classification.component.html'
})

@Injectable()
export class ClassificationComponent {

  URL_PATH: string;
  jsonUrl: any;
  classification: Classification;
  randomTweet: RandomTweet;

  constructor (
    private httpClient: HttpClient,
    private constantService: ConstantsService) {
    this.constantService.getJSON().subscribe(
      (data) => {
        this.jsonUrl = data;
        this.URL_PATH =  this.jsonUrl.url ;
      }
    );
  }

  postText(text: string): void {
  this.classification = null
    this.httpClient.post(this.URL_PATH + 'api',
    {
      text: text,
    })
    .subscribe(
      (response: any) => {
        console.log(response);
        this.classification = response
      }
    );
  }

  getProducts() {
      this.httpClient.get(this.URL_PATH + 'getRandomTweet')
      .subscribe(
        (response: any) => {
          console.log(response);
          this.randomTweet = response;
        }
      );
  }


  getSentiment(value: string): string{
    if(value){
        switch(value){
          case 'Positive':
            return 'positive'
          case 'Negative':
            return 'negative'
          case 'Neutral':
            return 'neutral'
          default:
            return ''
        }
    }else{
      return ''
    }
  }
}
