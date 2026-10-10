use serde::{Deserialize, Serialize};

pub(crate) const DEFAULT_REMOTE_START_COMMAND: &str =
    "$HOME/braintoken/agent/scripts/start_neurotoken_runtime.sh";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NeurotokenSshSettings {
    pub(crate) host: String,
    pub(crate) lang_graph_local_port: u16,
    pub(crate) lang_graph_remote_port: u16,
    pub(crate) coding_local_port: u16,
    pub(crate) coding_remote_port: u16,
    #[serde(default)]
    pub(crate) remote_start_command: Option<String>,
}

impl NeurotokenSshSettings {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.host.trim().is_empty() {
            return Err("SSH host is required.".to_string());
        }
        if self.host.contains(['\n', '\r', '\0']) || self.host.starts_with('-') {
            return Err("SSH host contains unsupported characters.".to_string());
        }
        if self.lang_graph_local_port == self.coding_local_port {
            return Err("LangGraph and coding local ports must be different.".to_string());
        }
        if let Some(command) = self.remote_start_command.as_deref() {
            if command.contains(['\n', '\r', '\0']) {
                return Err("Remote start command must be a single command line.".to_string());
            }
        }
        Ok(())
    }

    pub(crate) fn start_command(&self) -> &str {
        self.remote_start_command
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_REMOTE_START_COMMAND)
    }
}

pub(crate) fn availability_args() -> Vec<String> {
    vec!["-V".to_string()]
}

pub(crate) fn remote_start_args(settings: &NeurotokenSshSettings) -> Result<Vec<String>, String> {
    settings.validate()?;
    Ok(vec![
        "-T".to_string(),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        "ConnectTimeout=15".to_string(),
        "-o".to_string(),
        "RequestTTY=no".to_string(),
        "-o".to_string(),
        "ControlMaster=no".to_string(),
        "-o".to_string(),
        "ControlPath=none".to_string(),
        settings.host.trim().to_string(),
        settings.start_command().to_string(),
    ])
}

pub(crate) fn tunnel_args(settings: &NeurotokenSshSettings) -> Result<Vec<String>, String> {
    settings.validate()?;
    Ok(vec![
        "-N".to_string(),
        "-T".to_string(),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        "ControlMaster=no".to_string(),
        "-o".to_string(),
        "ControlPath=none".to_string(),
        "-o".to_string(),
        "ExitOnForwardFailure=yes".to_string(),
        "-o".to_string(),
        "ServerAliveInterval=15".to_string(),
        "-o".to_string(),
        "ServerAliveCountMax=3".to_string(),
        "-L".to_string(),
        format!(
            "127.0.0.1:{}:127.0.0.1:{}",
            settings.lang_graph_local_port, settings.lang_graph_remote_port
        ),
        "-L".to_string(),
        format!(
            "127.0.0.1:{}:127.0.0.1:{}",
            settings.coding_local_port, settings.coding_remote_port
        ),
        settings.host.trim().to_string(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> NeurotokenSshSettings {
        NeurotokenSshSettings {
            host: "brain-server".to_string(),
            lang_graph_local_port: 2024,
            lang_graph_remote_port: 2024,
            coding_local_port: 4732,
            coding_remote_port: 4732,
            remote_start_command: None,
        }
    }

    #[test]
    fn builds_one_tunnel_with_two_loopback_forwards() {
        let args = tunnel_args(&settings()).expect("tunnel args");
        assert_eq!(args.iter().filter(|arg| arg.as_str() == "-L").count(), 2);
        assert!(args.contains(&"127.0.0.1:2024:127.0.0.1:2024".to_string()));
        assert!(args.contains(&"127.0.0.1:4732:127.0.0.1:4732".to_string()));
        assert!(args.contains(&"ExitOnForwardFailure=yes".to_string()));
        assert!(args.contains(&"ServerAliveCountMax=3".to_string()));
    }

    #[test]
    fn uses_config_host_and_fixed_default_start_command() {
        let args = remote_start_args(&settings()).expect("start args");
        assert_eq!(args[args.len() - 2], "brain-server");
        assert_eq!(
            args.last().map(String::as_str),
            Some(DEFAULT_REMOTE_START_COMMAND)
        );
    }

    #[test]
    fn rejects_ambiguous_or_option_like_hosts_and_port_collisions() {
        let mut invalid = settings();
        invalid.host = "-oProxyCommand=bad".to_string();
        assert!(invalid.validate().is_err());
        invalid = settings();
        invalid.host = "host\nother".to_string();
        assert!(invalid.validate().is_err());
        invalid = settings();
        invalid.coding_local_port = invalid.lang_graph_local_port;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn rejects_multiline_remote_commands() {
        let mut invalid = settings();
        invalid.remote_start_command = Some("first\nsecond".to_string());
        assert!(invalid.validate().is_err());
    }
}
