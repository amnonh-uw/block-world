using System.Collections.Generic;
using UnityEngine;

public class FingerCollision : MonoBehaviour
{
	public Tray trayscript;

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
		if (trayscript.Ready)
			trayscript.CollisionHappened = true;
	}
}
