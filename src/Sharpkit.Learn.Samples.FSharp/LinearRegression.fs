module LinearRegressionSamples
open System
open Sharpkit.Learn
open Sharpkit.Learn.LinearModel

// MathNet.Numerics.Control.LinearAlgebraProvider <- (new Mkl.MklLinearAlgebraProvider() :> ILinearAlgebraProvider)

let clf = new LinearRegression()
clf.Fit(array2D [[0.0; 0.0]; [1.0; 1.0]; [2.0; 2.0]], [|0.0; 1.0; 2.0|]) |> ignore
Console.WriteLine(clf.Coef)

let prediction = clf.Predict([|3.0; 3.0|]);
Console.WriteLine(prediction);
