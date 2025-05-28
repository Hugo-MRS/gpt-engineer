from __future__ import annotations

import uuid

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gpt_engineer.applications.cli.cli_agent import CliAgent
from gpt_engineer.core.default.disk_execution_env import DiskExecutionEnv
from gpt_engineer.core.default.disk_memory import DiskMemory
from gpt_engineer.core.default.file_store import FileStore
from gpt_engineer.core.default.paths import PREPROMPTS_PATH, memory_path
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt


@dataclass
class Project:
    path: Path
    memory: DiskMemory
    execution_env: DiskExecutionEnv
    agent: CliAgent
    files: FileStore


PROJECTS_DIR = Path("projects")

projects: dict[str, Project] = {}
app = FastAPI()


class ProjectCreate(BaseModel):
    prompt: str


class IterateRequest(BaseModel):
    prompt: str


def _create_agent(path: Path) -> Project:
    memory = DiskMemory(memory_path(path))
    execution_env = DiskExecutionEnv()
    agent = CliAgent.with_default_config(
        memory,
        execution_env,
        preprompts_holder=PrepromptsHolder(PREPROMPTS_PATH),
    )
    files = FileStore(path)
    return Project(path, memory, execution_env, agent, files)


@app.post("/projects")
def create_project(req: ProjectCreate):
    project_id = str(uuid.uuid4())
    path = PROJECTS_DIR / project_id
    path.mkdir(parents=True, exist_ok=True)
    project = _create_agent(path)
    files_dict = project.agent.init(Prompt(req.prompt))
    project.files.push(files_dict)
    projects[project_id] = project
    return {"project_id": project_id, "files": dict(files_dict)}


@app.post("/projects/{project_id}/iterate")
def iterate_project(project_id: str, req: IterateRequest):
    project = projects.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    files_dict = project.files.pull()
    files_dict = project.agent.improve(files_dict, Prompt(req.prompt))
    project.files.push(files_dict)
    return {"files": dict(files_dict)}


@app.get("/projects/{project_id}/files")
def list_files(project_id: str):
    project = projects.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"files": dict(project.files.pull())}


@app.get("/projects/{project_id}/files/{file_path:path}")
def get_file(project_id: str, file_path: str):
    project = projects.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    full_path = project.path / file_path
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return {"content": full_path.read_text()}


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
