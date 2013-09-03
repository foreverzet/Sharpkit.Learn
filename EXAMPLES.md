Examples
===============
### [Linear Regression] (#linearregression)

<a id="linearregression"></a>Linear Regression
------------------

###F#
```F#
open System
open Sharpkit.Learn.LinearModel

let clf = new LinearRegression()
clf.Fit(array2D [[0.0; 0.0]; [1.0; 1.0]; [2.0; 2.0]], [|0.0; 1.0; 2.0|]) |> ignore
Console.Write(clf.Coef)
```

###C#
```C#
var clf = new Sharpkit.Learn.LinearModel.LinearRegression();
clf.Fit(new double[,] {{0, 0}, {1, 1}, {2, 2}}, new double[] {0, 1, 2});
Console.Write(clf.Coef.ToString());
```
