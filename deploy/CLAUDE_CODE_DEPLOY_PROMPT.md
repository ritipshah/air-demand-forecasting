You are helping me deploy a university project to AWS EC2. I'm a student in a
Virtualization & Cloud Computing (VCC) course, so explain each step in 1–2 plain
sentences as you go — I have to present this deployment.

PROJECT
- Name: "Short-Term Air Passenger Demand Forecasting via Adaptive Holt's Linear Trend Method"
- A Python Streamlit dashboard, containerised with Docker.
- Repo: <PASTE GITHUB REPO URL HERE>
- Contract the repo follows: Dockerfile at repo root; app listens on port 8501;
  health check at /_stcore/health returns "ok"; env var PROJECT_COURSE=vcc makes
  the app show our VCC team (Riti Shah, Pradhi Kanani, Dhruv Parekh).
- If the repo has deploy/DEPLOY.md, read it first and prefer its instructions.

GROUND RULES
- Region: ap-south-1 (Mumbai) unless I say otherwise.
- Stay inside the AWS Free Tier. Before creating ANYTHING that may cost money
  (instance type, Elastic IP / public IPv4, EBS above free size), tell me the
  likely cost and wait for my yes.
- Never ask me to paste AWS access keys or secrets into chat. If credentials are
  missing, tell me to run `aws configure` myself.
- Never commit .pem files, keys or resource-ID files to git; add them to .gitignore.
- Detect my OS (Windows / macOS / Linux) and adapt commands (e.g. .pem permissions).
- Tag every resource Project=air-demand-forecasting.

PART 1 — INFRASTRUCTURE
1. Check prerequisites: `aws --version` (v2), `aws sts get-caller-identity`, git.
2. Pick a free-tier instance type using
   `aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true`
   (prefer t3.micro or t2.micro, x86_64). Show me what you found.
3. Get the latest Amazon Linux 2023 AMI from SSM parameter
   /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64.
4. Create a key pair "airdemand-key", save the .pem locally with correct permissions.
5. Create security group "airdemand-sg": SSH (22) only from my current public IP
   (https://checkip.amazonaws.com), HTTP (80) from 0.0.0.0/0.
6. Launch ONE instance with a 20 GB gp3 root volume and user-data that:
   installs docker + git, enables docker, adds ec2-user to the docker group,
   and creates a 2 GB swap file (1 GB RAM is tight for building the image).
7. Wait until status checks pass, then SSH in and confirm `docker --version` works.
8. Save all resource IDs (instance, SG, key name, public IP, AMI) to
   deploy/aws-resources.env (gitignored) and print a summary table.

PART 2 — DEPLOY THE APP
1. On the instance: git clone the repo into ~/app.
2. `docker build -t airdemand:latest .`
3. `docker run -d --name airdemand --restart unless-stopped -p 80:8501 -e PROJECT_COURSE=vcc airdemand:latest`
4. Verify: `curl http://<public-ip>/_stcore/health` returns "ok", and give me the
   http:// URL to open in a browser.
5. Create deploy/redeploy.sh (SSH in → git pull → rebuild → replace container)
   so we can push updates with one command.

PART 3 — FOR OUR REPORT + SAFETY
1. Write deploy/DEPLOYMENT_LOG.md: architecture (User → Internet → Security Group
   → EC2 → Docker container → Streamlit), every step with the command used,
   resource IDs (no secrets), and a checklist of screenshots to take
   (EC2 console, security group rules, `docker ps`, the live app).
2. Create deploy/teardown.sh that stops/terminates the instance and deletes the
   SG, key pair and any Elastic IP — do NOT run it, just create it.
3. Offer (ask first) to set up an AWS Budget alert at $1 to my email, and
   optionally a GitHub Actions workflow that redeploys on every push to main.

Start with Part 1, step 1. Stop and ask me whenever a decision costs money.
