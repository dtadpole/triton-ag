mlflow:
	mlflow server --host localhost --port 5050

kbEval:
	uv run kbEvalRemoteServer.py 

codeRunServer:
	mcp dev codeRunServer.py
