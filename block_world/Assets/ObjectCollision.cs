using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectCollision : MonoBehaviour
{
    public Tray trayscript;
    bool FirstCollision = true;

	// Use this for initialization
	void Start ()
    {
		
	}
	
	// Update is called once per frame
	void Update ()
    {
		
	}

    void OnCollisionEnter(Collision col)
    {
        if (FirstCollision)
        {
            FirstCollision = false;
            trayscript.DropNextObject();
        }

		if (transform.position.y < 0) {
			// object has fallen below the tray
			trayscript.ObjList.Remove(gameObject);
			Object.Destroy (gameObject);
		}

		if (trayscript.Ready)
			trayscript.CollisionHappened = true;
    }
}
