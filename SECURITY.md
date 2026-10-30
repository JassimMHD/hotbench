# Security policy

## Supported versions

Until 1.0, only the latest `0.x` release gets security fixes.

## Reporting a vulnerability

Please **don't open a public issue.** Use GitHub's private reporting instead:
**Security → Report a vulnerability** on this repository. If that doesn't work
for you, email **jassimjr10you@gmail.com** with "Hotbench security" in the subject.

Include what you found, how to reproduce it, and the version
(`hotbench --version`). You will get a reply within 5 days and a fix or a plan
within 30 days. We will credit you in the release notes unless you'd rather
stay anonymous.

## What counts

Hotbench runs models and opens cameras on your machine, so these are in scope:

- the web UI reachable from other machines while bound to `127.0.0.1`;
- a recipe or model file that runs code nobody asked for. Pickle-based formats
  (`.pt`) can run code when loaded: Hotbench must warn before loading one from
  an untrusted source;
- credentials (camera passwords, API keys) leaking into the page, logs, run
  files or `--share` output;
- path traversal through the UI or the API.

Out of scope: running Hotbench with `--host 0.0.0.0` on an untrusted network
(the CLI warns about this), and bugs in the models or libraries themselves.
Report those upstream.
