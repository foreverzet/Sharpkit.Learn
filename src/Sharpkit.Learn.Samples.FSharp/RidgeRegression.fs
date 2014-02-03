module RidgeSamples
open System
open Sharpkit.Learn
open Sharpkit.Learn.LinearModel

let clf = new RidgeRegression(alpha = 0.5)
clf.Fit(array2D [[0.0; 0.0]; [0.0; 0.0]; [1.0; 1.0]], [|0.0; 0.1; 1.0|]) |> ignore
Console.WriteLine(clf.Coef)
Console.WriteLine(clf.Intercept)

let prediction = clf.Predict([|5.0; 6.0|])
Console.WriteLine(prediction);
