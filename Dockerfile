FROM python:3.10-slim

# OS dependencies
RUN apt update -y && apt upgrade -y
COPY packages.txt /tmp/os-tmp/
RUN xargs apt-get -y install < /tmp/os-tmp/packages.txt && rm -rf /tmp/os-tmp

# Add non-root user
RUN adduser --disabled-login worker
WORKDIR /home/worker
ENV PATH="/home/worker/.local/bin:${PATH}"

# `pip` is upgraded before using a worker user, because it’s installed as root and can’t be 
# accessed by a non-root user.
RUN pip install --upgrade pip

# Copy files.
COPY --chown=worker:worker . .

# Install dependencies as non-root user.
USER worker
RUN pip install --user -r requirements.txt




