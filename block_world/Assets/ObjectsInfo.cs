using System;
using System.Collections;
using System.Net;
using System.Collections.Generic;
using UnityEngine;

namespace Utilities
{
    [System.Serializable]
    class ObjectsInfo {
        public List<ObjectInfo> ObjList = new List<ObjectInfo> ();
        public ObjectInfo finger;
        public ObjectInfo target;
        public ObjectInfo main_camera;

        public ObjectsInfo(Tray t) {
            finger = new ObjectInfo (t.finger);
            target = new ObjectInfo (t.target);
            main_camera = new ObjectInfo (Camera.main.gameObject);

            foreach (GameObject go in t.ObjList)
                ObjList.Add (new ObjectInfo(go));
        }

        public string ToJson() {
            string s = JsonUtility.ToJson (this);
            return s;
        }

        public void Dump() {
            Debug.LogFormat ("finger {0}", finger.ToString ());
            Debug.LogFormat ("target {0}",  target.ToString());
            Debug.LogFormat ("camera {0}",  main_camera.ToString());
            foreach (ObjectInfo info in ObjList)
                Debug.LogFormat ("object {0}",  info.ToString());
        }
    }
}
