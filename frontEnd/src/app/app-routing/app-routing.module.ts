import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { ClassificationComponent } from './../product/product.component';
import { AppComponent } from './../app.component';
import { ProjectDetailsComponent } from './../project-details/project-details.component';

const routes: Routes = [
  { path: '', component: ProjectDetailsComponent },
  { path: 'tool', component: ClassificationComponent },
  { path: '**', component: ProjectDetailsComponent }

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})

export class AppRoutingModule { }

export const routingComponents = [ClassificationComponent];
