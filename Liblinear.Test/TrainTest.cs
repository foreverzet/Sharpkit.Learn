// -----------------------------------------------------------------------
// <copyright file="TrainTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Liblinear.Test
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    [TestClass]
    public class TrainTest {


    [TestMethod]
    public void testParseCommandLine() {
        Train train = new Train();


        foreach (SolverType solver in SolverType.values()) {
            train.parse_command_line(new [] {"-B", "5.3", "-s", "" + solver.getId(), "-p", "0.01", "model-filename"});
            Parameter param = train.Parameter;
            Assert.AreEqual(solver, param.solverType);
            // check default eps
            if (solver.getId() == 0 || solver.getId() == 2 //
                || solver.getId() == 5 || solver.getId() == 6) {
                Assert.AreEqual(0.01, param.eps);
            } else if (solver.getId() == 7) {
                Assert.AreEqual(0.1, param.eps);
            } else if (solver.getId() == 11) {
                Assert.AreEqual(0.001, param.eps);
            } else {
                Assert.AreEqual(0.1, param.eps);
            }
            // check if bias is set
            Assert.AreEqual(5.3, train.Bias);
            Assert.AreEqual(0.01, param.p);
        }
    }


    [TestMethod]
    // https://github.com/bwaldvogel/liblinear-java/issues/4
    public void testParseWeights() {
        Train train = new Train();
        train.parse_command_line(new [] {"-v", "10", "-c", "10", "-w1", "1.234", "model-filename"});
        Parameter parameter = train.Parameter;
        Assert.IsTrue(new[] {1}.SequenceEqual(parameter.weightLabel));
        Assert.IsTrue(new[] { 1.234 }.SequenceEqual(parameter.weight));

        train.parse_command_line(new [] {"-w1", "1.234", "-w2", "0.12", "-w3", "7", "model-filename"});
        parameter = train.Parameter;
        Assert.IsTrue(new [] {1, 2, 3}.SequenceEqual(parameter.weightLabel));
        Assert.IsTrue(new[] { 1.234, 0.12, 7 }.SequenceEqual(parameter.weight));
    }


    [TestMethod]
    public void testReadProblem()
    {
        var file = Path.GetTempFileName();

        var lines = new List<string>();
        lines.Add("1 1:1  3:1  4:1   6:1");
        lines.Add("2 2:1  3:1  5:1   7:1");
        lines.Add("1 3:1  5:1");
        lines.Add("1 1:1  4:1  7:1");
        lines.Add("2 4:1  5:1  7:1");

        File.WriteAllLines(file, lines);

        Train train = new Train();
        train.readProblem(file);


        Problem prob = train.Problem;
        Assert.AreEqual(1, prob.bias);
        Assert.AreEqual(lines.Count, prob.y.Length);
        Assert.IsTrue(new double[] {1, 2, 1, 1, 2}.SequenceEqual(prob.y));
        Assert.AreEqual(8, prob.n);
        Assert.AreEqual(prob.y.Length, prob.l);
        Assert.AreEqual(prob.y.Length, prob.x.Length);
       
        foreach (Feature[] nodes in prob.x)
        {
            Assert.IsTrue(nodes.Length <= prob.n);
            foreach (Feature node in nodes) {
                // bias term
                if (prob.bias >= 0 && nodes[nodes.Length - 1] == node) {
                    Assert.AreEqual(prob.n, node.Index);
                    Assert.AreEqual(prob.bias, node.Value);
                } else {
                    Assert.IsTrue(node.Index < prob.n);
                }
            }
        }
    }


    /**
     * unit-test for Issue #1 (http://github.com/bwaldvogel/liblinear-java/issues#issue/1)
     */
    [TestMethod]
    public void testReadProblemEmptyLine()
    {
        var file = Path.GetTempFileName();

        var lines = new List<string>();
        lines.Add("1 1:1  3:1  4:1   6:1");
        lines.Add("2 ");
        File.WriteAllLines(file, lines);


        Problem prob = Train.readProblem(new FileInfo(file), -1.0);
        Assert.AreEqual(-1, prob.bias);
        Assert.AreEqual(lines.Count, prob.y.Length);
        Assert.IsTrue(new double[] {1, 2}.SequenceEqual(prob.y));
        Assert.AreEqual(6, prob.n);
        Assert.AreEqual(prob.y.Length, prob.l);
        Assert.AreEqual(prob.y.Length, prob.x.Length);

        Assert.AreEqual(4, prob.x[0].Length);
        Assert.AreEqual(0, prob.x[1].Length);

        File.Delete(file);
    }


    [TestMethod]
    [ExpectedException(typeof(InvalidInputDataException))]
    public void testReadUnsortedProblem() {
        var file = Path.GetTempFileName();
        var lines = new List<String>();
        lines.Add("1 1:1  3:1  4:1   6:1");
        lines.Add("2 2:1  3:1  5:1   7:1");
        lines.Add("1 3:1  5:1  4:1"); // here's the mistake: not correctly sorted


        File.WriteAllLines(file, lines);

        Train train = new Train();
        train.readProblem(file);

        File.Delete(file);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidInputDataException))]
    public void testReadProblemWithInvalidIndex() {
        var file = Path.GetTempFileName();

        var lines = new List<string>();
        lines.Add("1 1:1  3:1  4:1   6:1");
        lines.Add("2 2:1  3:1  5:1  -4:1");

        File.WriteAllLines(file, lines);

        Train train = new Train();
        try 
        {
            train.readProblem(file);
        } catch (InvalidInputDataException e)
        {
            throw e;
        }

        File.Delete(file);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidInputDataException))]
    public void testReadWrongProblem()
    {
        var file = Path.GetTempFileName();

        var lines = new List<String>();
        lines.Add("1 1:1  3:1  4:1   6:1");
        lines.Add("2 2:1  3:1  5:1   7:1");
        lines.Add("1 3:1  5:a"); // here's the mistake: incomplete line

        File.WriteAllLines(file, lines);

        Train train = new Train();
        try {
            train.readProblem(file);
        } catch (InvalidInputDataException e) {
            throw e;
        }

        File.Decrypt(file);
    }
}

}
