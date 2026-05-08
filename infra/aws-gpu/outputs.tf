output "instance_id" {
  value       = aws_instance.gpu.id
  description = "EC2 instance ID."
}

output "public_ip" {
  value       = aws_instance.gpu.public_ip
  description = "Public IPv4 address for SSH."
}

output "ssh_private_key_path" {
  value       = local_sensitive_file.ssh_key_pem.filename
  description = "Local path to generated private key."
  sensitive   = true
}

output "ssh_command" {
  value       = "ssh -i ${local_sensitive_file.ssh_key_pem.filename} ${var.ssh_user}@${aws_instance.gpu.public_ip}"
  description = "Convenient SSH command."
}

output "remote_workdir" {
  value       = var.remote_workdir
  description = "Remote working directory used by run-cutile-smoke.sh."
}

output "ssh_cidr_effective" {
  value       = local.ssh_cidr
  description = "CIDR currently allowed for SSH ingress."
}

output "ssh_user" {
  value       = var.ssh_user
  description = "Preferred SSH login user."
}
