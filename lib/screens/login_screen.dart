// lib/screens/login_screen.dart
// Visora — Login & Signup screen (Admin-configurable, dynamic fields)
// Hindi UI messages, Admin JSON editor included (no hardcoded final data)
//
// Usage:
// Navigator.push(context, MaterialPageRoute(builder: (_) => LoginScreen(onSuccess: (user){ /* open home */ })));

import 'dart:convert';
import 'dart:ui';
import 'package:blur/blur.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';

typedef AuthCallback = void Function(Map<String, dynamic> userData);

class LoginScreen extends StatefulWidget {
  final AuthCallback? onSuccess;
  const LoginScreen({Key? key, this.onSuccess}) : super(key: key);

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  // Admin-controlled config (default). Admin can edit JSON using Admin panel.
  Map<String, dynamic> _config = {
    "fields": ["email", "password"], // show input controls in order
    "welcomeText": "Visora में स्वागत है — लॉग इन या नया अकाउंट बनाएं",
    "allowAdminLogin": true,
    "showSocialLogin": true,
    "requirePhoneInSignup": false,
    "minPasswordLength": 6
  };

  // controllers for dynamic fields
  final Map<String, TextEditingController> _controllers = {};
  final Map<String, String?> _errors = {};

  bool _isLoginMode = true; // toggle Login / Signup
  bool _loading = false;
  bool _adminOpen = false;
  final TextEditingController _adminJsonController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // init controllers for possible fields
    for (final f in ['name', 'email', 'password', 'phone', 'country']) {
      _controllers[f] = TextEditingController();
      _errors[f] = null;
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) c.dispose();
    _adminJsonController.dispose();
    super.dispose();
  }

  // Validate fields based on config
  bool _validate() {
    bool ok = true;
    _errors.updateAll((key, value) => null);

    final fields = List<String>.from(_config['fields'] ?? []);

    for (final f in fields) {
      final txt = _controllers[f]?.text.trim() ?? '';
      if (f == 'email') {
        if (txt.isEmpty || !txt.contains('@')) {
          _errors[f] = 'कृपया मान्य ईमेल डालें';
          ok = false;
        }
      } else if (f == 'password') {
        final minLen = (_config['minPasswordLength'] is int) ? _config['minPasswordLength'] as int : 6;
        if (txt.length < minLen) {
          _errors[f] = 'पासवर्ड कम से कम $minLen अक्षर का होना चाहिए';
          ok = false;
        }
      } else if (f == 'phone') {
        if (txt.isEmpty || txt.length < 6) {
          _errors[f] = 'कृपया मान्य फोन नंबर डालें';
          ok = false;
        }
      } else {
        if (_isLoginMode == false && f == 'name') {
          if (txt.isEmpty) {
            _errors[f] = 'नाम आवश्यक है';
            ok = false;
          }
        }
      }
    }

    setState(() {});
    return ok;
  }

  // Simulated auth – replace with real HTTP in future.
  Future<void> _performAuth() async {
    if (!_validate()) return;

    setState(() => _loading = true);

    try {
      // simulate network delay
      await Future.delayed(const Duration(seconds: 1));

      // build user object from fields (dynamic)
      final user = <String, dynamic>{};
      for (final key in _controllers.keys) {
        final v = _controllers[key]!.text.trim();
        if (v.isNotEmpty) user[key] = v;
      }
      user['plan'] = 'Free';
      user['credits'] = 100;
      user['id'] = 'u-${DateTime.now().millisecondsSinceEpoch}';

      // simulate login/signup difference
      final action = _isLoginMode ? 'login' : 'signup';
      // TODO: call backend here and handle token, errors etc.

      await Future.delayed(const Duration(milliseconds: 700));

      // success — call callback if provided
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('सफलता: $action पूरा हुआ')));
      widget.onSuccess?.call(user);
      // optionally navigate away; leaving to caller
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Auth failed: $e')));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  // Admin JSON editor
  void _openAdminEditor() {
    _adminJsonController.text = const JsonEncoder.withIndent('  ').convert(_config);
    setState(() => _adminOpen = true);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.85,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (context, sc) {
          return Blur(
            blur: 8,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: const BoxDecoration(color: Color(0xFF0F0F12), borderRadius: BorderRadius.vertical(top: Radius.circular(16))),
              child: ListView(
                controller: sc,
                children: [
                  Row(
                    children: [
                      Expanded(child: Text('Admin: Login Config (JSON)', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.w700))),
                      IconButton(onPressed: () => Navigator.of(context).pop(), icon: const Icon(Icons.close, color: Colors.white70))
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text('यहां से आप कौन-कौन से फ़ील्ड दिखेंगे और behavior बदल सकते हैं।', style: const TextStyle(color: Colors.white70)),
                  const SizedBox(height: 12),
                  SizedBox(
                    height: 420,
                    child: TextField(
                      controller: _adminJsonController,
                      keyboardType: TextInputType.multiline,
                      maxLines: null,
                      style: const TextStyle(color: Colors.white, fontFamily: 'monospace', fontSize: 13),
                      decoration: InputDecoration(
                        hintText: '{ "fields": ["email","password"], "welcomeText": "..."}',
                        hintStyle: const TextStyle(color: Colors.white38),
                        filled: true,
                        fillColor: Colors.white10,
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      ElevatedButton(
                        onPressed: () {
                          try {
                            final parsed = json.decode(_adminJsonController.text) as Map<String, dynamic>;
                            setState(() {
                              _config = parsed;
                            });
                            Navigator.of(context).pop();
                            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Config applied')));
                          } catch (e) {
                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('JSON parse error: $e')));
                          }
                        },
                        child: const Text('Apply'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
                    ],
                  )
                ],
              ),
            ),
          );
        },
      ),
    ).whenComplete(() => setState(() => _adminOpen = false));
  }

  // UI helpers
  Widget _animatedBackground() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF833AB4), Color(0xFFFF5E62), Color(0xFFFCCF31)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
    );
  }

  Widget _card() {
    final welcome = _config['welcomeText']?.toString() ?? 'Welcome';
    final fields = List<String>.from(_config['fields'] ?? ['email', 'password']);
    final showSocial = (_config['showSocialLogin'] == true);

    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Blur(
          blur: 6,
          borderRadius: BorderRadius.circular(18),
          child: Container(
            width: double.infinity,
            constraints: const BoxConstraints(maxWidth: 520),
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 18),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(18)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Center(child: LottieBuilderWidget()),
                const SizedBox(height: 6),
                Text(welcome, style: GoogleFonts.poppins(fontSize: 16, color: Colors.white70), textAlign: TextAlign.center),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(child: ElevatedButton(
                      onPressed: () => setState(() => _isLoginMode = true),
                      style: ElevatedButton.styleFrom(backgroundColor: _isLoginMode ? Colors.deepPurpleAccent : Colors.white10),
                      child: const Text('Login'),
                    )),
                    const SizedBox(width: 8),
                    Expanded(child: ElevatedButton(
                      onPressed: () => setState(() => _isLoginMode = false),
                      style: ElevatedButton.styleFrom(backgroundColor: !_isLoginMode ? Colors.deepPurpleAccent : Colors.white10),
                      child: const Text('Signup'),
                    )),
                  ],
                ),
                const SizedBox(height: 12),
                // dynamic fields
                ...fields.map((f) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8.0),
                    child: _buildField(f),
                  );
                }).toList(),
                // additional signup-only field (phone) if config requires
                if (!_isLoginMode && (_config['requirePhoneInSignup'] == true))
                  Padding(padding: const EdgeInsets.only(bottom: 8.0), child: _buildField('phone')),

                const SizedBox(height: 8),
                ElevatedButton(
                  onPressed: _loading ? null : _performAuth,
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.purpleAccent, padding: const EdgeInsets.symmetric(vertical: 14)),
                  child: _loading ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : Text(_isLoginMode ? 'Login' : 'Create Account'),
                ),
                const SizedBox(height: 8),
                if (showSocial)
                  Column(children: [
                    const Text('या', style: TextStyle(color: Colors.white54)),
                    const SizedBox(height: 8),
                    Row(children: [
                      Expanded(child: ElevatedButton.icon(onPressed: () => _showMsg('Google sign-in (placeholder)'), icon: const Icon(Icons.g_mobiledata), label: const Text('Google'), style: ElevatedButton.styleFrom(backgroundColor: Colors.white10))),
                      const SizedBox(width: 8),
                      Expanded(child: ElevatedButton.icon(onPressed: () => _showMsg('Apple sign-in (placeholder)'), icon: const Icon(Icons.apple), label: const Text('Apple'), style: ElevatedButton.styleFrom(backgroundColor: Colors.white10))),
                    ])
                  ]),
                const SizedBox(height: 8),
                TextButton(onPressed: _openAdminEditor, child: const Text('Admin Edit Config', style: TextStyle(color: Colors.white54))),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField(String key) {
    final labelMap = {
      'email': 'Email / ईमेल',
      'password': 'Password / पासवर्ड',
      'name': 'Full name / पूरा नाम',
      'phone': 'Phone / फ़ोन',
      'country': 'Country / देश'
    };
    final isPassword = key == 'password';
    final ctrl = _controllers[key]!;
    final error = _errors[key];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(labelMap[key] ?? key, style: const TextStyle(color: Colors.white70)),
        const SizedBox(height: 6),
        TextField(
          controller: ctrl,
          obscureText: isPassword,
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            hintText: labelMap[key],
            hintStyle: const TextStyle(color: Colors.white38),
            filled: true,
            fillColor: Colors.white10,
            errorText: error,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
          ),
        ),
      ],
    );
  }

  void _showMsg(String s) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(s)));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          _animatedBackground(),
          SafeArea(
            child: Center(child: _card()),
          ),
        ],
      ),
    );
  }
}

// small helper widget for Lottie with fallback
class LottieBuilderWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    try {
      return SizedBox(width: 120, height: 120, child: Lottie.asset('assets/lottie/loading.json', repeat: true));
    } catch (_) {
      return CircleAvatar(radius: 36, backgroundColor: Colors.white12, child: Icon(Icons.person, color: Colors.white));
    }
  }
}
