variable "aws_region" {
  description = "AWS region for the GPU instance."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile name used by Terraform."
  type        = string
}

variable "instance_type" {
  description = "GPU instance type. g5.xlarge = 1x A10G (sm_86), 4 vCPU, 16 GiB. cuTile requires sm_80+."
  type        = string
  default     = "g5.xlarge"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. CUDA 13.2 + LLVM 21 + Rust toolchain needs ~30 GiB; 100 leaves headroom."
  type        = number
  default     = 100
}

variable "vpc_id" {
  description = "Optional explicit VPC ID. If null, Terraform looks up the default VPC."
  type        = string
  default     = null
}

variable "subnet_id" {
  description = "Optional explicit subnet ID. If null, Terraform picks the first subnet in the selected VPC."
  type        = string
  default     = null
}

variable "project_name" {
  description = "Tag prefix for created resources."
  type        = string
  default     = "cutile-rs"
}

variable "remote_workdir" {
  description = "Remote directory on the EC2 host where the cutile-rs repo is cloned."
  type        = string
  default     = "/home/ubuntu/cutile-rs"
}

variable "ssh_user" {
  description = "Preferred SSH user for the AMI (auto-fallback also exists in helper scripts)."
  type        = string
  default     = "ubuntu"
}

variable "ssh_cidr" {
  description = "Optional CIDR allowed to SSH. If null, uses your current public IP/32."
  type        = string
  default     = null
}

variable "ami_id" {
  description = "Optional explicit AMI ID. If null, use DLAMI SSM parameter."
  type        = string
  default     = null
}

variable "dlami_ssm_parameter" {
  description = "SSM parameter path for AWS DLAMI GPU image. Default tracks Ubuntu 24.04 (the OS the cutile-rs README is tested against)."
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}
