provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

data "aws_vpc" "default" {
  count   = var.vpc_id == null ? 1 : 0
  default = true
}

data "aws_subnets" "default_vpc_subnets" {
  count = var.subnet_id == null ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [var.vpc_id != null ? var.vpc_id : data.aws_vpc.default[0].id]
  }
}

data "http" "my_ip" {
  count = var.ssh_cidr == null ? 1 : 0
  url   = "https://checkip.amazonaws.com"
}

data "aws_ssm_parameter" "dlami" {
  count = var.ami_id == null ? 1 : 0
  name  = var.dlami_ssm_parameter
}

locals {
  run_id    = formatdate("YYYYMMDD-hhmmss", timestamp())
  vpc_id    = var.vpc_id != null ? var.vpc_id : data.aws_vpc.default[0].id
  subnet_id = var.subnet_id != null ? var.subnet_id : sort(data.aws_subnets.default_vpc_subnets[0].ids)[0]
  ssh_cidr  = var.ssh_cidr != null ? var.ssh_cidr : "${trimspace(data.http.my_ip[0].response_body)}/32"
  ami_id    = var.ami_id != null ? var.ami_id : data.aws_ssm_parameter.dlami[0].value
  key_name  = "${var.project_name}-cutile-${local.run_id}"
  key_path  = "${path.module}/.ssh/${local.key_name}.pem"
  common_tags = {
    Project = var.project_name
    Role    = "cutile_smoke"
    Managed = "terraform"
  }
}

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "local_sensitive_file" "ssh_key_pem" {
  filename        = local.key_path
  content         = tls_private_key.ssh.private_key_pem
  file_permission = "0600"
}

resource "aws_key_pair" "generated" {
  key_name   = local.key_name
  public_key = tls_private_key.ssh.public_key_openssh
  tags       = local.common_tags
}

resource "aws_security_group" "ssh" {
  name_prefix = "${var.project_name}-cutile-"
  description = "SSH access for temporary cuTile GPU host"
  vpc_id      = local.vpc_id

  ingress {
    description = "SSH from local machine"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_instance" "gpu" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.generated.key_name
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.ssh.id]
  associate_public_ip_address = true

  instance_initiated_shutdown_behavior = "terminate"

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = file("${path.module}/userdata.sh")

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-cutile-gpu"
  })
}
