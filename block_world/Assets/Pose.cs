using System;
using System.Collections;
using System.Net;
using System.Collections.Generic;
using UnityEngine;

namespace Utilities
{
    class Pose
    {
        public Vector3 position;
        public Vector3 rotation;

        public Pose(string s)
        {
            ElementSplitter es = new ElementSplitter (s);

            position = es.GetNext(Vector3.zero);
            rotation = es.GetNext(Vector3.zero);
        }

        public override string ToString()
        {
            return "postion: " + this.position.ToString () + " rotation: " + this.rotation.ToString ();
        }
    }
}
