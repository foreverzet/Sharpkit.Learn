Examples
===============
*  [Linear Regression] (#linearregression)
*  [Ridge Regression] (#ridgeregression)
*  [Ridge Classifier] (#ridgeclassifier)
*  [Logistic Regression] (#logisticregression)

<a id="linearregression"></a>Linear Regression
------------------
Please find full documentation here:[Ordinary Least Squares] (http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

F\#
```F#
open System
open Sharpkit.Learn.LinearModel

let clf = new LinearRegression()
clf.Fit(array2D [[0.0; 0.0]; [1.0; 1.0]; [2.0; 2.0]], [|0.0; 1.0; 2.0|]) |> ignore
Console.WriteLine(clf.Coef)

let prediction = clf.Predict([|3.0; 3.0|]);
Console.WriteLine(prediction);
```

C\#
```C#
// Learn
var clf = new Sharpkit.Learn.LinearModel.LinearRegression();
clf.Fit(new double[,] {{0, 0}, {1, 1}, {2, 2}}, new double[] {0, 1, 2});
Console.WriteLine(clf.Coef.ToString());

// Predict
var prediction = clf.Predict(new double[] {3, 3});
Console.WriteLine(prediction);
```


<a id="ridgeregression"></a>Ridge Regression
------------------
Please find full documentation here:[Ridge Regression] (http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

F\#
```F#
open System
open Sharpkit.Learn.LinearModel

let clf = new RidgeRegression(alpha = 0.5)
clf.Fit(array2D [[0.0; 0.0]; [0.0; 0.0]; [1.0; 1.0]], [|0.0; 0.1; 1.0|]) |> ignore
Console.WriteLine(clf.Coef)
Console.WriteLine(clf.Intercept)

let prediction = clf.Predict([|5.0; 6.0|])
Console.WriteLine(prediction);
```

C\#
```C#
var clf = new RidgeRegression(alpha: 0.5);
clf.Fit(new[,] {{0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}}, new[] {0.0, 0.1, 1.0});
Console.WriteLine(clf.Coef);
Console.WriteLine(clf.Intercept);

var prediction = clf.Predict(new[] {5.0, 6.0});
Console.WriteLine(prediction);
```

<a id="ridgeclassifier"></a>Ridge Classifier
------------------
C\#
```C#
var clf = new RidgeClassifier<string>(alpha: 0.5);
clf.Fit(new[,] { { 0.0, 0.0 }, { 0.0, 0.0 }, { 1.0, 1.0 } }, new[] { "a", "b", "c" });
Console.WriteLine(clf.Coef);
Console.WriteLine(clf.Intercept);

var prediction = clf.Predict(new[] { 5.0, 6.0 });
Console.WriteLine(prediction);
```

<a id="logisticregression"></a>Logistic Regression
------------------
Please find full documentation here:[Logistic Regression] (http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
