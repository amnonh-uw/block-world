using System;
using System.Threading;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;

namespace Utilities
{
    public class Listener 
    {
        public HttpListenerResponse Response;

        HttpListener listener;
        Thread listenerThread;
        bool exitListener;
        string listenerAction;
        string listenerArgs;
		string listenerBody;
        Object listenerLock;

        public Listener()
        {
            listener = new HttpListener();
            string port = System.Environment.GetEnvironmentVariable("PORT");
            if (port == null)
                port = "9000";


            string prefix = "http://*:" + port + "/";
            listener.Prefixes.Add(prefix);
			listenerThread = null;
            listenerAction = null;
			listenerArgs = null;
			listenerAction = null;
			listenerBody = null;
        }

        public bool Start()
        {
            try
            {
                listener.Start();
            }
            catch (SocketException  ex)
            {
                return false;
            }


            listenerThread = new Thread(listenerMain);
            exitListener = false;
            listenerLock = new Object();
            listenerThread.Start();

            return true;
        }

    	void listenerMain()
    	{
        	while(!exitListener)
        	{
            	HttpListenerContext context = listener.GetContext();
            	HttpListenerRequest request = context.Request;
            	Response = context.Response;

				if (request.HttpMethod == "GET") {
					Response.StatusCode = 200;
					Response.StatusDescription = "OK";
					Response.Close ();
				}


				if (request.HasEntityBody) {
					Stream body = request.InputStream;
					Encoding encoding = request.ContentEncoding;
					StreamReader reader = new StreamReader (body, encoding);

                	if (request.ContentType.ToLower() == "application/x-www-form-urlencoded")
                	{
                    	string s = reader.ReadToEnd();
                    	

                    	lock (listenerLock)
                    	{
							listenerAction = null;
							listenerArgs = null;
							listenerBody = s;

							string[] pairs = s.Split('&');

							for (int i = 0; i < pairs.Length; i++) {
								string s2 = UrlDecode(pairs[i]);
								// listenerBody += " url after decode " + s2;
								
								string[] items = s2.Split(new char[] {'='}, 2);
                            	string name = items[0];
                            	string value = items[1];

								if (name == "command") {
									listenerAction = value;
									// listenerBody += " found command " + value;
								}

								if (name == "args") {
									listenerArgs = value;
									// listenerBody += " found args " + value;
								}
                        	}
                    	}
                	}

                	body.Close();
                	reader.Close();
            	}
        	}
    	}

    	public void Stop() {
        	exitListener = true;
			if (listener != null)
				listener.Abort ();
			
			if (listenerThread != null)
				listenerThread.Abort ();
    	}

		public void GetAction(out string action, out string args, out string body)
    	{
        	lock (listenerLock)
        	{
            	action = listenerAction;
            	args = listenerArgs;
				body = listenerBody;

            	listenerAction = null;
            	listenerArgs = null;
				listenerBody = null;
        	}
    	}

		public static string UrlDecode (string str) 
		{
			return UrlDecode(str, Encoding.UTF8);
		}

		static char [] GetChars (MemoryStream b, Encoding e)
		{
			return e.GetChars (b.GetBuffer (), 0, (int) b.Length);
		}

		static void WriteCharBytes (IList buf, char ch, Encoding e)
		{
			if (ch > 255) {
				foreach (byte b in e.GetBytes (new char[] { ch }))
					buf.Add (b);
			} else
				buf.Add ((byte)ch);
		}

		public static string UrlDecode (string str, Encoding e)
		{
			if (null == str) 
				return null;

			if (str.IndexOf ('%') == -1 && str.IndexOf ('+') == -1)
				return str;

			if (e == null)
				e = Encoding.UTF8;

			long len = str.Length;
			var bytes = new List <byte> ();
			int xchar;
			char ch;

			for (int i = 0; i < len; i++) {
				ch = str [i];
				if (ch == '%' && i + 2 < len && str [i + 1] != '%') {
					if (str [i + 1] == 'u' && i + 5 < len) {
						// unicode hex sequence
						xchar = GetChar (str, i + 2, 4);
						if (xchar != -1) {
							WriteCharBytes (bytes, (char)xchar, e);
							i += 5;
						} else
							WriteCharBytes (bytes, '%', e);
					} else if ((xchar = GetChar (str, i + 1, 2)) != -1) {
						WriteCharBytes (bytes, (char)xchar, e);
						i += 2;
					} else {
						WriteCharBytes (bytes, '%', e);
					}
					continue;
				}

				if (ch == '+')
					WriteCharBytes (bytes, ' ', e);
				else
					WriteCharBytes (bytes, ch, e);
			}

			byte[] buf = bytes.ToArray ();
			bytes = null;
			return e.GetString (buf);

		}

		static int GetChar (byte [] bytes, int offset, int length)
		{
			int value = 0;
			int end = length + offset;
			for (int i = offset; i < end; i++) {
				int current = GetInt (bytes [i]);
				if (current == -1)
					return -1;
				value = (value << 4) + current;
			}

			return value;
		}

		static int GetChar (string str, int offset, int length)
		{
			int val = 0;
			int end = length + offset;
			for (int i = offset; i < end; i++) {
				char c = str [i];
				if (c > 127)
					return -1;

				int current = GetInt ((byte) c);
				if (current == -1)
					return -1;
				val = (val << 4) + current;
			}

			return val;
		}

		static int GetInt (byte b)
		{
			char c = (char) b;
			if (c >= '0' && c <= '9')
				return c - '0';

			if (c >= 'a' && c <= 'f')
				return c - 'a' + 10;

			if (c >= 'A' && c <= 'F')
				return c - 'A' + 10;

			return -1;
		}
	}
}