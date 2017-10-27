using System;
using System.Collections;
using System.Net;
using System.Collections.Generic;
using UnityEngine;

namespace Utilities
{
    class ElementSplitter {
        string[] elems;
        int index;

        public ElementSplitter(string s) {
            index = 0;
            s = s.Replace (" ", string.Empty);
            s = s.Replace ("(", string.Empty);
            s = s.Replace (")", string.Empty);

            elems = s.Split (',');
        }

        public float GetNext(float def)
        {
            if (index < elems.Length) {
                string s = elems [index++].Trim();
                if (s == string.Empty)
                    return def;
                return System.Convert.ToSingle (s);
            } else
                return def;
        }

        public Vector3 GetNext(Vector3 def) {
            return new Vector3 (GetNext (def.x), GetNext (def.y), GetNext (def.z));
        }

        public Vector4 GetNext(Vector4 def) {
            return new Vector4 (GetNext (def.x), GetNext (def.y), GetNext (def.z), GetNext (def.w));
        }

        public Quaternion GetNext(Quaternion def) {
            return new Quaternion (GetNext (def.x), GetNext (def.y), GetNext (def.z), GetNext (def.w));
        }
    }
}
